src/runtime.h: src/main.rs .cargo/cbindgen.toml
	cbindgen --config .cargo/cbindgen.toml src/main.rs -o src/runtime.h
