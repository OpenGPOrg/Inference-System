BUILD_DIR := build
NPROC     := $(shell nproc 2>/dev/null || echo 4)

.PHONY: all build clean distclean

all: build

build:
	@if [ ! -f "$(BUILD_DIR)/CMakeCache.txt" ]; then \
		cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release; \
	fi
	cmake --build $(BUILD_DIR) -j$(NPROC)

# Clean ONLY OpenGP compiled objects/libraries/binaries; preserve CMake build files & vendor GGML/CUDA
clean:
	@for d in $(BUILD_DIR)/CMakeFiles/opengp_*.dir; do \
		if [ -d "$$d" ]; then find "$$d" -name "*.o" -delete; fi; \
	done
	rm -f $(BUILD_DIR)/libopengp_*.a \
	      $(BUILD_DIR)/opengp_perception \
	      $(BUILD_DIR)/opengp_reasoner \
	      $(BUILD_DIR)/opengp_reasoning \
	      $(BUILD_DIR)/opengp_infer

# Full wipe including compiled vendor GGML/CUDA libraries
distclean:
	rm -rf $(BUILD_DIR)
