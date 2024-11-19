use commitlog::{LogOptions, CommitLog, Offset};
use serde::{Serialize, Deserialize};
use std::path::Path;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum LogEntry {
    Propose {
        txid: String,
        timestamp: u64,
    },
    Vote {
        txid: String,
        participant_id: u32,
        vote: bool,
    },
    Decision {
        txid: String,
        commit: bool,
    },
    ParticipantState {
        participant_id: u32,
        txid: String,
        prepared: bool,
    }
}

pub struct FaultTolerantLog {
    log: CommitLog,
}

impl FaultTolerantLog {
    pub fn new(path: &str) -> Self {
        let opts = LogOptions::new(path);
        let log = CommitLog::new(opts).expect("Failed to create commit log");
        Self { log }
    }

    pub fn append(&mut self, entry: LogEntry) -> Result<Offset, std::io::Error> {
        let bytes = bincode::serialize(&entry).expect("Failed to serialize log entry");
        self.log.append(&bytes)
    }

    pub fn read_all(&self) -> Vec<LogEntry> {
        let mut entries = Vec::new();
        let mut offset = 0;
        
        while let Ok(Some(record)) = self.log.read(offset) {
            if let Ok(entry) = bincode::deserialize(&record.payload) {
                entries.push(entry);
            }
            offset = record.offset + 1;
        }
        entries
    }
} 