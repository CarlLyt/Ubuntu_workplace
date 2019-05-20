################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../project/src/build/CMakeFiles/3.5.1/CompilerIdCXX/CMakeCXXCompilerId.cpp 

OBJS += \
./project/src/build/CMakeFiles/3.5.1/CompilerIdCXX/CMakeCXXCompilerId.o 

CPP_DEPS += \
./project/src/build/CMakeFiles/3.5.1/CompilerIdCXX/CMakeCXXCompilerId.d 


# Each subdirectory must supply rules for building sources it contributes
project/src/build/CMakeFiles/3.5.1/CompilerIdCXX/%.o: ../project/src/build/CMakeFiles/3.5.1/CompilerIdCXX/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/usr/include/opencv -I/home/carllee/workspace/ConfrimFlat/project/include -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


