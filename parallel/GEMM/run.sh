#!/bin/bash

for (( t=1 ; t<=$1 ; t*=2));
do
    echo "Running run_gemm.jl with $t threads"
    julia --threads=$t run_gemm.jl --ncpu $t
done
