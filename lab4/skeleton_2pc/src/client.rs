//!
//! client.rs
//! Implementation of 2PC client
//!
extern crate ipc_channel;
extern crate log;
extern crate stderrlog;

use std::thread;
use std::time::Duration;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::collections::HashMap;

use client::ipc_channel::ipc::IpcReceiver as Receiver;
use client::ipc_channel::ipc::IpcSender as Sender;

use message;
use message::MessageType;
use message::RequestStatus;

// Client state and primitives for communicating with the coordinator
#[derive(Debug)]
pub struct Client {
    pub id_str: String,
    pub running: Arc<AtomicBool>,
    pub num_requests: u32,
    successful_ops: u64,
    failed_ops: u64,
    unknown_ops: u64,
    sender: Sender<message::ProtocolMessage>,
    receiver: Receiver<message::ProtocolMessage>,
    transaction_status: HashMap<String, RequestStatus>,
}

///
/// Client Implementation
/// Required:
/// 1. new -- constructor
/// 2. pub fn report_status -- Reports number of committed/aborted/unknown
/// 3. pub fn protocol(&mut self, n_requests: i32) -- Implements client side protocol
///
impl Client {

    ///
    /// new()
    ///
    /// Constructs and returns a new client, ready to run the 2PC protocol
    /// with the coordinator.
    ///
    /// HINT: You may want to pass some channels or other communication
    ///       objects that enable coordinator->client and client->coordinator
    ///       messaging to this constructor.
    /// HINT: You may want to pass some global flags that indicate whether
    ///       the protocol is still running to this constructor
    ///
    pub fn new(id_str: String,
               running: Arc<AtomicBool>,
               num_requests: u32,
               sender: Sender<message::ProtocolMessage>,
               receiver: Receiver<message::ProtocolMessage>) -> Client {
        Client {
            id_str: id_str,
            running: running,
            num_requests: num_requests,
            sender,
            receiver,
            failed_ops: 0,
            successful_ops: 0,
            unknown_ops: 0,
            transaction_status: HashMap::new(),
        }
    }

    ///
    /// wait_for_exit_signal(&mut self)
    /// Wait until the running flag is set by the CTRL-C handler
    ///
    pub fn wait_for_exit_signal(&mut self) {
        trace!("{}::Waiting for exit signal", self.id_str.clone());

        // TODO
        while self.running.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_millis(100));
        }

        self.shutdown();
        trace!("{}::Exiting", self.id_str.clone());
    }

    ///
    /// send_next_operation(&mut self)
    /// Send the next operation to the coordinator
    ///
    pub fn send_next_operation(&mut self) {
        if !self.running.load(Ordering::Relaxed) {
            return;
        }

        // Create a new request with a unique TXID.
        self.num_requests += 1;
        let txid = format!("{}_op_{}", self.id_str.clone(), self.num_requests);
        let pm = message::ProtocolMessage::generate(message::MessageType::ClientRequest,
                                                    txid.clone(),
                                                    self.id_str.clone(),
                                                    self.num_requests);
        // Send the message using the IPC channel
        match self.sender.send(pm) {
            Ok(_) => {
                info!("{}::Sent operation #{}", self.id_str.clone(), self.num_requests);
            },
            Err(e) => {
                error!("{}::Failed to send operation: {}", self.id_str.clone(), e);
                self.transaction_status.insert(txid.clone(), RequestStatus::Unknown);
                self.shutdown();
            }
        }
    }

    ///
    /// recv_result()
    /// Wait for the coordinator to respond with the result for the
    /// last issued request. Note that we assume the coordinator does
    /// not fail in this simulation
    ///

    pub fn recv_result(&mut self) {
        match self.receiver.recv() {
            Ok(msg) => {
                match msg.mtype {
                    MessageType::ClientResultCommit => {
                        self.transaction_status.insert(msg.txid.clone(), RequestStatus::Committed);
                        self.successful_ops += 1;
                        
                    }
                    MessageType::ClientResultAbort => {
                        self.transaction_status.insert(msg.txid.clone(), RequestStatus::Aborted);
                        self.failed_ops += 1;
                    }
                    MessageType::CoordinatorExit => {
                        info!("{}::Coordinator is shutting down", self.id_str.clone());
                        self.shutdown();
                    }
                    unexpected_msg => {
                        self.unknown_ops += 1;
                        error!("{}::Unexpected message type {:?} received for operation: {}", 
                                self.id_str.clone(), unexpected_msg, msg.txid);  
                    }
                }
            },
            Err(e) => {
                error!("{}::Connection to coordinator lost: {:?}", self.id_str.clone(), e);
                
            }
        }
        
    }

    ///
    /// report_status()
    /// Report the abort/commit/unknown status (aggregate) of all transaction
    /// requests made by this client before exiting.
    ///
    pub fn report_status(&mut self) {
        let successful_ops: u64 = self.successful_ops;
        let failed_ops: u64 = self.failed_ops;
        let unknown_ops: u64 = self.unknown_ops;

        println!("{:16}:\tCommitted: {:6}\tAborted: {:6}\tUnknown: {:6}", 
        self.id_str.clone(), successful_ops, failed_ops, unknown_ops);
    }

    ///
    /// protocol()
    /// Implements the client side of the 2PC protocol
    /// HINT: if the simulation ends early, don't keep issuing requests!
    /// HINT: if you've issued all your requests, wait for some kind of
    ///       exit signal before returning from the protocol method!
    ///
    pub fn protocol(&mut self, n_requests: u32) {
        //println!("{}::Starting protocol with {} requests", self.id_str, n_requests);
        
        // Continue with normal protocol 
        for i in 0..n_requests {
            if !self.running.load(Ordering::Relaxed) {
                println!("{}::Protocol stopped early at request {}/{}", self.id_str, i, n_requests);
                break;
            }
            self.send_next_operation();
            self.recv_result();
        }

        //println!("{}::All requests processed, waiting for exit signal", self.id_str);
        self.wait_for_exit_signal();
        //self.report_status();
    }


    pub fn start(&mut self) {
        self.protocol(self.num_requests);
    }

    
    pub fn shutdown(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        info!("{}::Shutting down client", self.id_str.clone());
    }

}
