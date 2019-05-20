################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CXX_SRCS += \
../project/src/build/CMakeFiles/feature_tests.cxx 

C_SRCS += \
../project/src/build/CMakeFiles/feature_tests.c 

CXX_DEPS += \
./project/src/build/CMakeFiles/feature_tests.d 

OBJS += \
./project/src/build/CMakeFiles/feature_tests.o 

C_DEPS += \
./project/src/build/CMakeFiles/feature_tests.d 


# Each subdirectory must supply rules for building sources it contributes
project/src/build/CMakeFiles/%.o: ../project/src/build/CMakeFiles/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Compiler'
	gcc -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

project/src/build/CMakeFiles/%.o: ../project/src/build/CMakeFiles/%.cxx
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/usr/include/opencv -I/home/carllee/workspace/ConfrimFlat/project/include -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


