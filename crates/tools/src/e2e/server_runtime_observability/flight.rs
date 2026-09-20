use arrow_flight::{
    Action, FlightInfo,
    sql::{
        ActionCancelQueryRequest, ActionCancelQueryResult, Any, ProstMessageExt, TicketStatementQuery,
        client::FlightSqlServiceClient,
    },
};
use prost::Message as _;
use tonic::transport::Channel;

use super::E2eResult;

pub(super) const CANCEL_RESULT_CANCELLING: i32 = 2;

pub(super) fn query_id_from_statement_flight_info(flight_info: &FlightInfo) -> E2eResult<String> {
    let ticket = flight_info
        .endpoint
        .first()
        .and_then(|endpoint| endpoint.ticket.as_ref())
        .ok_or("FlightInfo missing statement ticket")?;
    let any = Any::decode(&*ticket.ticket)?;
    let statement = any.unpack::<TicketStatementQuery>()?.ok_or("FlightInfo ticket is not TicketStatementQuery")?;
    Ok(String::from_utf8(statement.statement_handle.to_vec())?)
}

pub(super) async fn get_query_stats(channel: &Channel, query_id: &str) -> E2eResult<String> {
    action_payload(channel, "GetQueryStats", query_id.as_bytes().to_vec()).await
}

pub(super) async fn get_admission_stats(channel: &Channel) -> E2eResult<String> {
    action_payload(channel, "GetAdmissionStats", Vec::new()).await
}

async fn action_payload(channel: &Channel, action_type: &str, body: Vec<u8>) -> E2eResult<String> {
    let mut frames = action_frames(channel, action_type, body).await?;
    if frames.len() != 1 {
        return Err(format!("{action_type} returned {} results, expected exactly one", frames.len()).into());
    }
    Ok(String::from_utf8(frames.remove(0))?)
}

pub(super) async fn action_frames(channel: &Channel, action_type: &str, body: Vec<u8>) -> E2eResult<Vec<Vec<u8>>> {
    let mut client = FlightSqlServiceClient::new(channel.clone());
    let action = Action { r#type: action_type.to_string(), body: body.into() };
    let mut stream = client.do_action(action).await?;
    let mut frames = Vec::new();
    while let Some(result) = stream.message().await? {
        frames.push(result.body.to_vec());
    }
    if frames.is_empty() {
        return Err(format!("{action_type} returned no result").into());
    }
    Ok(frames)
}

pub(super) async fn cancel_query(channel: &Channel, flight_info: &FlightInfo) -> E2eResult<ActionCancelQueryResult> {
    let mut client = FlightSqlServiceClient::new(channel.clone());
    let request = ActionCancelQueryRequest { info: flight_info.encode_to_vec().into() };
    let action = Action { r#type: "CancelQuery".to_string(), body: request.as_any().encode_to_vec().into() };
    let mut stream = client.do_action(action).await?;
    let result = stream.message().await?.ok_or("CancelQuery returned no result")?;
    if stream.message().await?.is_some() {
        return Err("CancelQuery returned more than one result".into());
    }
    let any = Any::decode(&*result.body)?;
    any.unpack::<ActionCancelQueryResult>()?.ok_or_else(|| "CancelQuery returned wrong result type".into())
}
