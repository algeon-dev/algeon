use std::{
    collections::BTreeMap,
    fs::{self, File},
    os::unix::process::ExitStatusExt,
    path::{Path, PathBuf},
    process::{ExitStatus, Stdio},
    time::Duration,
};

use tokio::{
    process::{Child, Command},
    select, time,
};
use tonic::transport::Channel;

use super::E2eResult;
use crate::connection_retry::{ConnectionRetryConfig, connect_with_retry};

const CONNECT_BACKOFF: Duration = Duration::from_millis(100);

pub(super) struct ServerChild {
    child: Child,
}

pub(super) struct ServerSpawnConfig {
    pub(super) server_binary: PathBuf,
    pub(super) stdout_path: PathBuf,
    pub(super) stderr_path: PathBuf,
    pub(super) env: BTreeMap<String, String>,
}

#[derive(Debug, Clone)]
pub(super) struct ProcessExit {
    pub(super) success: bool,
    pub(super) code: Option<i32>,
    pub(super) signal: Option<i32>,
    pub(super) display: String,
}

impl ServerChild {
    pub(super) fn spawn_config(config: ServerSpawnConfig) -> E2eResult<Self> {
        ensure_parent(&config.stdout_path)?;
        ensure_parent(&config.stderr_path)?;
        let stdout = File::create(&config.stdout_path)?;
        let stderr = File::create(&config.stderr_path)?;
        let mut command = Command::new(&config.server_binary);
        command.stdout(Stdio::from(stdout)).stderr(Stdio::from(stderr)).kill_on_drop(true);
        // The server rejects settings of the removed Iceberg integration, and any inherited
        // ALGEON_SERVER_* value would change the topology under test.
        for (name, _) in std::env::vars() {
            if name.starts_with("ALGEON_SERVER_")
                || name.starts_with("ALGEON_ICEBERG_")
                || name.starts_with("ALGEON_LAKEHOUSE_")
                || name == "TOKIO_WORKER_THREADS"
            {
                command.env_remove(name);
            }
        }
        command.envs(config.env);
        Ok(Self { child: command.spawn()? })
    }

    pub(super) async fn connect_with_timeout(&mut self, url: &str, window: Duration) -> E2eResult<Channel> {
        let attempts = usize::try_from(window.as_millis() / CONNECT_BACKOFF.as_millis()).unwrap_or(usize::MAX).max(1);
        let connect = connect_with_retry(
            url,
            ConnectionRetryConfig::fixed_attempts(attempts, CONNECT_BACKOFF),
            |channel| channel,
            |error| format!("release server did not accept connections at {url}: {error:?}"),
        );
        // The retry helper cannot observe the child. A server that exits during startup would
        // otherwise surface only after the whole connect window.
        select! {
            result = connect => result,
            status = self.child.wait() => Err(format!(
                "release server exited before accepting Flight SQL connections: {}",
                status?
            )
            .into()),
        }
    }

    pub(super) async fn terminate(&mut self, signal: &str, wait: Duration) -> E2eResult<ProcessExit> {
        let Some(pid) = self.child.id() else {
            return Err("release server child has no process id".into());
        };
        let signal_status =
            std::process::Command::new("kill").args([format!("-{signal}"), pid.to_string()]).status()?;
        if !signal_status.success() {
            return Err(format!("failed to send SIG{signal} to release server pid {pid}: {signal_status}").into());
        }
        let status = time::timeout(wait, self.child.wait())
            .await
            .map_err(|_| format!("release server pid {pid} did not exit within {} seconds", wait.as_secs()))??;
        Ok(process_exit(status))
    }

    pub(super) fn process_id(&self) -> Option<u32> {
        self.child.id()
    }

    pub(super) fn require_running(&mut self, context: &str) -> E2eResult<()> {
        if let Some(status) = self.child.try_wait()? {
            return Err(format!("{context}: server exited unexpectedly: {}", process_exit(status).display).into());
        }
        Ok(())
    }
}

pub(super) fn ensure_parent(path: &Path) -> E2eResult<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn process_exit(status: ExitStatus) -> ProcessExit {
    ProcessExit { success: status.success(), code: status.code(), signal: status.signal(), display: status.to_string() }
}
