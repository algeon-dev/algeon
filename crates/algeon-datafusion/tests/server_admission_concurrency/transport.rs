use std::{io, net::SocketAddr, time::Duration};

use arrow_flight::{
    Action,
    sql::{Any, TicketStatementQuery, client::FlightSqlServiceClient},
};
use futures::TryStreamExt;
use prost::Message;
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel},
    task::JoinHandle,
    time::{sleep, timeout},
};
use tokio_util::sync::CancellationToken;
use tonic::transport::{Channel, Endpoint};

use crate::support::{ConcurrencyServer, ListenMode, MANAGED_CAPACITY_BYTES, QUERY_CAP_BYTES};

#[derive(Debug, PartialEq, Eq)]
enum TransportEvent {
    Ping,
    PingAcknowledged,
    ServerClosed,
}

/// Forwards HTTP/2 until cut, then discards bytes with both TCP sockets open.
/// A missing FIN/RST reproduces connection loss that requires a keepalive ping.
struct LossyConnection {
    addr: SocketAddr,
    cut: CancellationToken,
    events: UnboundedReceiver<TransportEvent>,
    task: JoinHandle<()>,
}

impl LossyConnection {
    async fn spawn(server_addr: SocketAddr) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind proxy");
        let addr = listener.local_addr().expect("proxy address");
        let cut = CancellationToken::new();
        let forwarding_cut = cut.clone();
        let (events_tx, events) = unbounded_channel();
        let task = tokio::spawn(async move {
            let (client, _) = listener.accept().await.expect("accept client");
            let server = TcpStream::connect(server_addr).await.expect("connect server");
            let (client_read, client_write) = client.into_split();
            let (server_read, server_write) = server.into_split();
            let _ = tokio::join!(
                forward_frames(client_read, server_write, &forwarding_cut, &events_tx, true),
                forward_frames(server_read, client_write, &forwarding_cut, &events_tx, false),
            );
        });
        Self { addr, cut, events, task }
    }

    async fn wait_for(&mut self, wanted: TransportEvent, deadline: Duration) {
        timeout(deadline, async {
            loop {
                let event = self.events.recv().await.expect("proxy remains active");
                if event == wanted {
                    return;
                }
                assert_ne!(event, TransportEvent::ServerClosed, "server closed before {wanted:?}");
            }
        })
        .await
        .unwrap_or_else(|_| panic!("transport did not report {wanted:?} within {deadline:?}"));
    }
}

impl Drop for LossyConnection {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn forward_frames(
    mut reader: impl AsyncRead + Unpin,
    mut writer: impl AsyncWrite + Unpin,
    cut: &CancellationToken,
    events: &UnboundedSender<TransportEvent>,
    from_client: bool,
) -> io::Result<()> {
    let mut buffer = [0; 16 * 1024];
    let mut pending = Vec::new();
    let mut preface_remaining = if from_client { 24 } else { 0 };
    loop {
        let bytes = match reader.read(&mut buffer).await {
            Ok(0) | Err(_) => {
                if !from_client {
                    let _ = events.send(TransportEvent::ServerClosed);
                }
                return Ok(());
            },
            Ok(bytes) => bytes,
        };
        if !cut.is_cancelled() {
            writer.write_all(&buffer[..bytes]).await?;
        }
        let preface_bytes = preface_remaining.min(bytes);
        preface_remaining -= preface_bytes;
        pending.extend_from_slice(&buffer[preface_bytes..bytes]);
        let mut consumed = 0;
        while pending.len() - consumed >= 9 {
            let header = &pending[consumed..consumed + 9];
            let payload_len = (usize::from(header[0]) << 16) | (usize::from(header[1]) << 8) | usize::from(header[2]);
            let frame_len = 9 + payload_len;
            if pending.len() - consumed < frame_len {
                break;
            }
            if header[3] == 6 && payload_len == 8 {
                let event = match (from_client, header[4] & 1 != 0) {
                    (false, false) => Some(TransportEvent::Ping),
                    (true, true) => Some(TransportEvent::PingAcknowledged),
                    _ => None,
                };
                if let Some(event) = event {
                    let _ = events.send(event);
                }
            }
            consumed += frame_len;
        }
        pending.drain(..consumed);
    }
}

async fn query_stats(client: &mut FlightSqlServiceClient<Channel>, query_id: &str) -> String {
    let action = Action { r#type: "GetQueryStats".to_string(), body: query_id.as_bytes().to_vec().into() };
    let mut stream = client.do_action(action).await.expect("query stats action");
    let result = stream.message().await.expect("query stats result").expect("one query stats result");
    String::from_utf8(result.body.to_vec()).expect("query stats UTF-8")
}

async fn lost_connection_releases_attempt(listen_mode: ListenMode) {
    let server =
        ConcurrencyServer::spawn_with_listener(4_000_000, MANAGED_CAPACITY_BYTES, QUERY_CAP_BYTES, listen_mode).await;
    let mut observer = server.client().await;
    let mut proxy = LossyConnection::spawn(server.addr).await;
    let channel = Endpoint::from_shared(format!("http://{}", proxy.addr))
        .expect("proxy endpoint")
        .connect()
        .await
        .expect("connect through proxy");
    let mut client = FlightSqlServiceClient::new(channel);
    let info = client
        .execute("select value from concurrency_rows order by value desc".to_string(), None)
        .await
        .expect("plan streaming query");
    let ticket = info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).expect("ticket");
    let any = Any::decode(&*ticket.ticket).expect("ticket protobuf");
    let query_ticket = any.unpack::<TicketStatementQuery>().expect("ticket type").expect("statement query");
    let query_id = String::from_utf8(query_ticket.statement_handle.to_vec()).expect("query id");
    let mut stream = client.do_get(ticket).await.expect("DoGet");
    assert!(stream.try_next().await.expect("first batch").expect("nonempty result").num_rows() > 0);
    server.wait_for_admission(1, 0).await;

    // The result exceeds the HTTP/2 window. Tonic continues acknowledging
    // pings while the application leaves its result stream unread.
    proxy.wait_for(TransportEvent::Ping, Duration::from_secs(45)).await;
    proxy.wait_for(TransportEvent::PingAcknowledged, Duration::from_secs(5)).await;
    let report = server.backend.device_ledger_reports().expect("ledger report").remove(0);
    assert_eq!(report.active_attempts, 1, "an acknowledging consumer keeps its backpressured attempt live");
    assert_eq!(report.live_grants_bytes, QUERY_CAP_BYTES);

    while proxy.events.try_recv().is_ok() {}
    proxy.cut.cancel();
    proxy.wait_for(TransportEvent::Ping, Duration::from_secs(45)).await;
    proxy.wait_for(TransportEvent::ServerClosed, Duration::from_secs(15)).await;

    // Neither CancelQuery nor shutdown runs here. The transport must drop
    // the root stream while the client still owns its open socket and body.
    let stats = timeout(Duration::from_secs(120), async {
        loop {
            let stats = query_stats(&mut observer, &query_id).await;
            if stats.contains("# sealed=true\n") {
                break stats;
            }
            sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("disconnected attempt seals after native cleanup");
    assert!(stats.contains("# outcome=cancelled\n"), "{stats}");
    assert!(stats.contains("# attempt_outcome=cancelled\n"), "{stats}");
    assert!(stats.contains("# attempt_capacity_disposition=released\n"), "{stats}");
    assert_eq!(stats.lines().filter(|line| line.starts_with("# attempt_id=")).count(), 1, "{stats}");
    server.wait_for_admission(0, 0).await;
    let report = server.backend.device_ledger_reports().expect("released ledger report").remove(0);
    assert!(report.grants.is_empty(), "{report:?}");
    assert_eq!(report.live_grants_bytes, 0);
    assert_eq!(report.available_bytes, MANAGED_CAPACITY_BYTES);

    drop(stream);
    drop(client);
    drop(proxy);
    drop(observer);
    server.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn addr_keepalive_drops_a_lost_root_stream_and_releases_its_attempt_grant() {
    lost_connection_releases_attempt(ListenMode::Addr).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn prebound_keepalive_drops_a_lost_root_stream_and_releases_its_attempt_grant() {
    lost_connection_releases_attempt(ListenMode::Prebound).await;
}
