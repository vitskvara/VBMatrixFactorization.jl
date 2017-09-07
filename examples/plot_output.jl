# prints the classification result produce by running mil_data.jl
# as an input, it takes the folder with saved outputs and possibly a number
# of an output file that only needs to be read
# run from the command line, e.g.
# julia path/to/file/plot_output.jl path/with/outputs path/to/save - this plots all the files in output path
# and saves them to given folder
# or
# julia path/to/file/plot_output.jl path/with/outputs path/to/save 1 3 7- this plots first, third and seventh file in the folder
# and saves them to given folder
using JLD
include("mil_util.jl")

output_path = ARGS[1]
if size(ARGS)[1] > 1
    save_path = ARGS[2]
    mkpath(save_path)
else
    save_path =""
end
files = readdir(output_path)
nfiles = size(files)[1]

if size(ARGS)[1] > 2
    inds = map(x->parse(Float64,x),ARGS[3:end])
else
    inds = 1:nfiles
end

for n in inds
    n = convert(Int, n)
    file = files[n]
    println("File n = $n")
    try
        data = load("$output_path/$file")
        plot_statistics(data, verb = false, save_path = save_path)    
    catch e
        println("Something bad happened, probably wrong file type.")
    end
    n+=1
end

#merge all the output files?
#if save_path != ""
    #try
        #cd(save_path)
        #run(pipeline(`gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -sOutputFile=out.pdf *.eps`))
        #println("All output files merged to out.pdf")
    #end
#end