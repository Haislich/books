set shell := ["powershell.exe", "-c"]

test:
	cargo build
	mdbook serve ./test_book