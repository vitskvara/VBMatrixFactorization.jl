# prints the classification result produce by running mil_data.jl
# as an input, it takes the folder with saved outputs and possibly a number
# of an output file that only needs to be read
# run from the command line, e.g.
# julia path/to/file/read_output.jl path/with/outputs - this reads all the files in output path
# or
# julia path/to/file/read_output.jl path/with/outputs 1 3 7- this reads first, third and seventh file in the folder
using JLD
include("mil_util.jl")

output_path = ARGS[1]
files = readdir(output_path)
nfiles = size(files)[1]

if size(ARGS)[1] > 1
    inds = map(x->parse(Float64,x),ARGS[2:end])
else
    inds = 1:nfiles
end

for n in inds
    n = convert(Int, n)
    file = files[n]
    println("File n = $n")
    try
        data = load("$output_path/$file")
        table_summary(data)    
    catch e
        println("Something bad happened, probably wrong file type.")
        println(e)
    end
    n+=1
end