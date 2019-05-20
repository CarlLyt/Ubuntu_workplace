################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../project/src/Camera.cpp \
../project/src/ConfrimFlat.cpp \
../project/src/main.cpp 

OBJS += \
./project/src/Camera.o \
./project/src/ConfrimFlat.o \
./project/src/main.o 

CPP_DEPS += \
./project/src/Camera.d \
./project/src/ConfrimFlat.d \
./project/src/main.d 


# Each subdirectory must supply rules for building sources it contributes
project/src/%.o: ../project/src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/usr/include/opencv -I/usr/local/lib -I/home/carllee/workspace/ConfrimFlat/project/include -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


