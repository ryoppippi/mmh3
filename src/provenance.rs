//! Where a generation came from, which a server writes beside what it generated: the checkpoints
//! it read, the build of mmh3 that read them and the machine that ran it, so that a picture whose
//! time or look is in question can be traced to what made it.

use serde_json::{Value, json};
use std::path::Path;

/// A checkpoint as `{file, size, digest}`, with the digest two machines compare checkpoints by,
/// or with nulls for what cannot be read of it.
pub fn model(path: &Path) -> Value {
    let size = std::fs::metadata(path).ok().map(|metadata| metadata.len());
    let digest = mmh3_core::safetensors::SafeTensors::open(path)
        .ok()
        .map(|file| format!("{:016x}", crate::worker::digest(&file)));
    json!({
        "file": path.file_name().map(|name| name.to_string_lossy()),
        "size": size,
        "digest": digest,
    })
}

/// This build of mmh3: its version and the commit it was built from, or null for a build from a
/// source without git.
pub fn build() -> Value {
    let commit = env!("MMH3_COMMIT");
    json!({
        "version": env!("CARGO_PKG_VERSION"),
        "commit": (!commit.is_empty()).then_some(commit),
    })
}

/// The machine, with what it says of its power at the time of asking: a Mac's speed depends on its
/// power mode and adapter as much as on its chip. The GPU's clock would say more, but only
/// powermetrics reads it, as root, so it is null.
pub fn machine() -> Value {
    json!({
        "model": sysctl("hw.model"),
        "chip": sysctl("machdep.cpu.brand_string"),
        "memory_bytes": sysctl("hw.memsize").and_then(|bytes| bytes.parse::<u64>().ok()),
        "power_mode": pmset(&["-g"]).as_deref().and_then(power_mode),
        "adapter_watts": pmset(&["-g", "adapter"]).as_deref().and_then(adapter_watts),
        "gpu_clock_mhz_median": Value::Null,
    })
}

fn sysctl(name: &str) -> Option<String> {
    command("sysctl", &["-n", name])
}

fn pmset(arguments: &[&str]) -> Option<String> {
    command("pmset", arguments)
}

fn command(program: &str, arguments: &[&str]) -> Option<String> {
    let output = std::process::Command::new(program)
        .args(arguments)
        .output()
        .ok()?;
    output.status.success().then_some(())?;
    let text = String::from_utf8(output.stdout).ok()?;
    let text = text.trim();
    (!text.is_empty()).then(|| text.to_owned())
}

/// The power mode of `pmset -g`'s `powermode` line, by name where it has one.
fn power_mode(settings: &str) -> Option<String> {
    let value = settings.lines().find_map(|line| {
        let mut words = line.split_whitespace();
        (words.next() == Some("powermode")).then(|| words.next())?
    })?;
    Some(
        match value {
            "0" => "automatic",
            "1" => "low",
            "2" => "high",
            other => other,
        }
        .to_owned(),
    )
}

/// The watts of `pmset -g adapter`'s `Wattage = 90W` line, or None without an adapter.
fn adapter_watts(adapter: &str) -> Option<u64> {
    adapter.lines().find_map(|line| {
        let value = line.trim().strip_prefix("Wattage")?.trim_start();
        value
            .strip_prefix('=')?
            .trim()
            .trim_end_matches('W')
            .parse()
            .ok()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_power_settings() {
        let settings = "System-wide power settings:\n Sleep On Power Button 1\n powernap             1\n powermode            2\n";
        assert_eq!(power_mode(settings).as_deref(), Some("high"));
        assert_eq!(power_mode(" lowpowermode 1\n"), None);
        let adapter = " Wattage = 90W\n Current = 4500mA\n Voltage = 20000mV\n";
        assert_eq!(adapter_watts(adapter), Some(90));
        assert_eq!(adapter_watts("No adapter attached.\n"), None);
    }
}
