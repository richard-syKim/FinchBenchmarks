#!/bin/bash

for (( t=1 ; t<=$1 ; t++));
do
	echo "Running lb_testing_shard.jl with $t threads"
	julia --threads=$t lb_testing_shard.jl --ncpu $t
done
