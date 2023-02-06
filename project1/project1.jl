using Graphs
using Printf
using CSV
using DataFrames
using SpecialFunctions
using Random

"""
    write_gph(dag::DiGraph, idx2names, filename)

Takes a DiGraph, a Dict of index to names and a output filename to write the graph in `gph` format.
"""
function write_gph(dag::DiGraph, idx2names, filename)
    open(filename, "w") do io
        for edge in edges(dag)
            @printf(io, "%s,%s\n", idx2names[src(edge)], idx2names[dst(edge)])
        end
    end
end

"""
Takes in the path of a file containing an edgelist and creates a SimpleDiGraph
from it. Returns the graph object and a dictionary mapping the name of a vertex
to its index in the graph.

Inspiration taken from Ed post #368.
"""
function load_gph(graphfile, datafile)
    vars = split(readline(datafile), ',')
    names_to_idx = Dict(vars[i] => i for i in eachindex(vars))
    g = SimpleDiGraph(length(vars))
    open(graphfile, "r") do f
        while ! eof(f)
            line = readline(f)
            source, dest = split(line, ',')            
            add_edge!(g, names_to_idx[source], names_to_idx[dest])
        end
    end

    return g, names_to_idx
end

"""
Takes in the path to a datafile (csv format), a dictionary mapping variable
names to vertex indices, and a graph and obtains the counts found in the dataset
for each value of each variable for each parental instantiation.

Inspiration taken from Algorithm 4.1 in the textbook.
"""
function load_counts(datafile, names_to_idx, g)
    data = Matrix(CSV.read(datafile, DataFrame))
    return get_counts_from_data(data, names_to_idx, g)
end

function get_counts_from_data(data, names_to_idx, g)
    n = length(names_to_idx) # number of variables
    r = [maximum(col) for col in eachcol(data)] # num values from max in dataset
    q = [prod([r[j] for j in inneighbors(g,i)]) for i in 1:n] # num parental instantiations    
    M = [zeros(q[i], r[i]) for i in 1:n] # for each variable, init parent x val

    for sample in eachrow(data) # for each sample
        for i in 1:n # for each variable
            k = sample[i] # variable value is the same as the value's index k
            parents = inneighbors(g,i)
            j = 1
            if !isempty(parents)
                # convert parent values (cartesian indices) to linear index
                # for parental instantiations
                lin = LinearIndices(Tuple(r[parents]))
                j = lin[sample[parents]...]
            end
            M[i][j, k] += 1
        end
    end
    return M
end

"""
Takes in counts from a dataset (array of n variables, each containing an array
of j parental instantiations by k variable values) and Dirichlet prior counts
(same shape as the dataset counts) and calculates the bayesian score. The score
is returned.
"""
function calculate_bayesian_score(M, priors)
    # for each combination of i, j, k
    score = sum(
        [
            sum(loggamma.(M[i] + priors[i])) - sum(loggamma.(priors[i]))
            for i in eachindex(M)
        ]
    )

    # for each combination of i, j, summing across k indices (e.g. m_ij0)
    score += sum(
        [
            (sum(loggamma.(sum(priors[i], dims = 2))
                - loggamma.(sum(M[i], dims = 2) + sum(priors[i], dims = 2))))
            for i in eachindex(M)
        ]
    )

    return score
end

"""
Takes in a file containing an edgelist defining a graph and a file containing
sample data and calculates the bayesian score (probability of the data given
the graph structure).
"""
function compute(graphfile, datafile)
    g, names_to_idx = load_gph(graphfile, datafile)
    counts = load_counts(datafile, names_to_idx, g)
    priors = [ones(size(var_counts)) for var_counts in counts] # uniform prior
    return calculate_bayesian_score(counts, priors)
end

function find_best_graph(datafile, searchmethod)
    # init data and mapping of variable to index from datafile
    data_df = CSV.read(datafile, DataFrame)
    data = Matrix(data_df)
    vars = names(data_df)
    names_to_idx = Dict(vars[i] => i for i in eachindex(vars))
    
    if searchmethod == "k2"
        var_order = shuffle(eachindex(vars)) # random topological ordering
        return run_k2_search(var_order, names_to_idx, data)
    end
end


# fn = ARGS[1]
fn = "learn"

if fn == "score"
    # graphfile = ARGS[2]
    # datafile = ARGS[3]
    # outputfile = ARGS[4]
    graphfile = "example\\example.gph"
    datafile = "example\\example.csv"
    outputfile = "test2.txt"
    
    score = compute(graphfile, datafile)
    open(outputfile, "w") do f
        write(f, string(score))
    end
elseif fn == "learn"
    # datafile = ARGS[2]
    # outputfile = ARGS[3]
    # searchmethod = ARGS[4]
    datafile = "example\\example.csv"
    outputfile = "test.txt"
    searchmethod = "k2"

    g, score = find_best_graph(datafile, searchmethod)

end
