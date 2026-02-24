//! CLI helper for generating Argon2id PHC strings to paste into the
//! Flight SQL server's `[auth.mode.basic_bearer.users].password_hash`
//! config field.
//!
//! Usage:
//!
//! ```bash
//! cargo run --features server --example hash_password -- '<password>'
//! ```
//!
//! Reads exactly one positional argument; refuses to operate on an empty
//! string. Prints the PHC string to stdout and nothing else, so it can
//! be safely captured with command substitution. Argon2 parameters are
//! the `argon2` crate defaults (m=19456, t=2, p=1) matching the current
//! OWASP recommendation for interactive logins.

#[cfg(feature = "server")]
fn main() -> std::process::ExitCode {
    use datafusion_cudf::server::auth::hash_password;

    let mut args = std::env::args().skip(1);
    let password = match args.next() {
        Some(p) if !p.is_empty() => p,
        _ => {
            eprintln!("usage: hash_password <password>");
            return std::process::ExitCode::from(2);
        },
    };
    if args.next().is_some() {
        eprintln!("usage: hash_password <password>  (exactly one argument)");
        return std::process::ExitCode::from(2);
    }
    match hash_password(&password) {
        Ok(phc) => {
            println!("{phc}");
            std::process::ExitCode::SUCCESS
        },
        Err(error) => {
            eprintln!("hash failed: {error}");
            std::process::ExitCode::FAILURE
        },
    }
}

#[cfg(not(feature = "server"))]
fn main() {
    eprintln!("hash_password requires `--features server`");
    std::process::exit(2);
}
