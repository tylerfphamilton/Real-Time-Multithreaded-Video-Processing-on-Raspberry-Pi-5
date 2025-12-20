CXX := g++						
OUT := main		
SRC := main.cpp thread_helpers.cpp gray_sobel_neon_filtering.cpp 

OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4)
OPENCV_LIBS   := $(shell pkg-config --libs opencv4)

CXXFLAGS := -Wall -Wextra -Werror -O3 -mcpu=cortex-a76 -mtune=cortex-a76 -ffast-math $(OPENCV_CFLAGS)
LDFLAGS  := $(OPENCV_LIBS) -lpthread

.PHONY: all clean

all: $(OUT)

$(OUT): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(OUT) $(LDFLAGS)

clean:
	rm -f $(OUT)
