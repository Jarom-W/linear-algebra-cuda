.PHONY: build run

build:
	cd ./build && cmake .. && make -j

run:
	cd ./build && python3 convert.py && explorer.exe output.png 
.PHONY: clean
clean:
	rm -rf ./build/*
