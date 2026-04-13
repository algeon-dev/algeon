fn main() {
    let outcome = xtask::parity::run(std::env::args_os());
    print!("{}", outcome.json);
    std::process::exit(outcome.exit_code);
}
