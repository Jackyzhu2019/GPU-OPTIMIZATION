CC := nvcc
CFLAGS := -O3 -arch=sm_89 -w -lineinfo
TARGET := mimo_64
SRCS := $(wildcard *.cu)
OBJS := $(SRCS:.cu=.o)
timestamp := $(shell date +"%Y%m%d_%H%M%S")
output_file := report_$(timestamp)
CUDA_PROF ?= 0
MIMO64_OPT ?= 1

ifeq ($(MIMO64_OPT),1)
CFLAGS += -DMIMO64_OPT=1
endif

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)

run: $(TARGET)
ifeq ($(CUDA_PROF), 0)
	taskset -c 24 ./$(TARGET)
endif
ifeq ($(CUDA_PROF), 1)
	nsys profile --stats=true ./$(TARGET)
endif
ifeq ($(CUDA_PROF), 2)
	ncu --set full -o $(output_file) ./$(TARGET)
endif

.PHONY: all clean run