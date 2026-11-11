use std::time::Duration;

use mmh3_studio::mock::{self, Config};

const USAGE: &str = "mock-server [--listen ADDR] [--step SECONDS] [--text SECONDS] [--decode SECONDS]

Serves the mmh3 server API for images on ADDR (127.0.0.1:8833 by default): six steps of
--step seconds each, then a painted PNG. A prompt containing [fail] fails after its steps.";

fn main() {
    let mut listen = "127.0.0.1:8833".to_owned();
    let mut config = Config::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| exit(&format!("{arg} needs a value")));
        let mut seconds = || Duration::from_secs_f64(value().parse().unwrap_or_else(|_| exit("expected seconds")));
        match arg.as_str() {
            "--listen" => listen = value(),
            "--step" => config.step = seconds(),
            "--text" => config.text = seconds(),
            "--decode" => config.decode = seconds(),
            "-h" | "--help" => {
                println!("{USAGE}");
                return;
            }
            other => exit(&format!("unknown argument {other}")),
        }
    }
    match mock::spawn(&listen, config) {
        Ok(addr) => println!("mock mmh3 server on http://{addr}"),
        Err(e) => exit(&format!("couldn't listen on {listen}: {e}")),
    }
    loop {
        std::thread::park();
    }
}

fn exit(message: &str) -> ! {
    eprintln!("{message}\n\n{USAGE}");
    std::process::exit(2)
}
