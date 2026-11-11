//! Records the commit mmh3 is built from, which a server writes beside every generation.

use std::path::Path;
use std::process::Command;

fn git(arguments: &[&str]) -> Option<String> {
    let output = Command::new("git").args(arguments).output().ok()?;
    output.status.success().then_some(())?;
    Some(String::from_utf8(output.stdout).ok()?.trim().to_owned())
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // HEAD moves on a checkout and the branch's ref on a commit, which is all that changes the
    // commit. A source without git, as a Nix build has, has no commit to say.
    let mut watched = Vec::new();
    if let Some(directory) = git(&["rev-parse", "--git-dir"]) {
        watched.push(format!("{directory}/HEAD"));
    }
    if let Some(common) = git(&["rev-parse", "--git-common-dir"]) {
        watched.push(format!("{common}/packed-refs"));
        if let Some(branch) = git(&["symbolic-ref", "-q", "HEAD"]) {
            watched.push(format!("{common}/{branch}"));
        }
    }
    for path in watched.iter().filter(|path| Path::new(path).exists()) {
        println!("cargo:rerun-if-changed={path}");
    }
    let commit = git(&["rev-parse", "HEAD"]).unwrap_or_default();
    println!("cargo:rustc-env=MMH3_COMMIT={commit}");
}
