module PDEHarness

using DrWatson
using JLD2
using Base: ImmutableDict
using TOML
using TimerOutputs
using ProgressMeter
using Printf: @sprintf

export integrate_stably, should_perform_io, mkplotpath, mksimpath, diagnostics_csv_path, frame_writeout, load_from_frame!, restart_from, logpath

include("utils.jl")

DrWatson.allignore(_::Dict{String, Any}) = ("##dict_is_normalized##",)

function normalize!(d::Dict)
    ensure!(d, "##dict_is_normalized##", true)
    return Base.ImmutableDict(pairs(d)...)
end

function is_normalized(d::Base.ImmutableDict)
    get(d, "##dict_is_normalized##", false)
end

"""
should_perform_io(sim) -> Bool

Whether we should perform writeouts and diagnostics.
For example, non-MPI root nodes 
"""
should_perform_io(_) = true

mysavename(d) = begin
    @assert is_normalized(d)
    newpairs = filter(p -> p.first != "problem_func", collect(pairs(d)))
    d2 = Base.ImmutableDict(newpairs...)
    string(hash(d2), base=16)
end

mksimpath(d::Base.ImmutableDict) = begin
    @assert is_normalized(d)
    p = datadir("sims", mysavename(d))
    mkpath(p)
    mkpath(joinpath(p, "logs"))
    params_path = joinpath(p, "params.toml")
    !isfile(params_path) && open(params_path, "w") do io
        TOML.print(io, d, sorted=true) do x
            x isa Symbol && return string(x)
            x isa NamedTuple && return "NamedTuple"
            return x
        end
    end
    p
end

mkplotpath(d) = begin
    @assert is_normalized(d)
    p = plotsdir(mysavename(d))
    mkpath(p)
    p
end

diagnostics_csv_path(d) = begin
    @assert is_normalized(d)
    p = mksimpath(d)
    joinpath(p, "diagnostics.csv")
end

function logpath(d, sim)
    p = mksimpath(d)
    joinpath(p, "logs", "output.log")
end

function diagnostics_initial(sim, d, runner::Function)
    csvfile = joinpath(mksimpath(d), "diagnostics.csv")
    rm(csvfile, force=true)

    output = runner(sim, 0.0)::NamedTuple
    table = [values(output)...]
    header = [keys(output)...]

    should_perform_io(sim) && open(csvfile, "w") do io
        write(io, join(header, ","))
        write(io, "\n")
        write(io, join(table, ","))
        write(io, "\n")
    end
end

function diagnostics_soln(sim, t, d, runner::Function)
    csvfile = joinpath(mksimpath(d), "diagnostics.csv")

    output = runner(sim, t)::NamedTuple
    table = [values(output)...]

    should_perform_io(sim) && open(csvfile, "a") do io
        write(io, join(table, ","))
        write(io, "\n")
    end
end

"""
frame_writeout(sim, t) -> Dict{String, Any}

Return the data to be written out at the current frame, as a dictionary with string keys.
"""
function frame_writeout(sim, t) 
    @warn "Default frame writeout is empty"
end

function writeout_initial(sim, d)
    datafile = joinpath(mksimpath(d), "writeouts.jld2")
    should_perform_io(sim) && rm(datafile, force=true)
    writeout_solution(sim, 0.0, d)
end

function writeout_solution(sim, t, d)
    datafile = joinpath(mksimpath(d), "writeouts.jld2")

    frame_data = frame_writeout(sim, t)

    should_perform_io(sim) || return

    jldopen(datafile, "a+") do file
        allkeys = Vector(keys(file))
        frames = filter(k -> startswith(k, "frame_"), allkeys)
        key = "frame_$(length(frames))"

        file[key] = frame_data
    end
end

function integrate_stably(step!, sim, t_end, d::Base.ImmutableDict; 
    initial_dt=0.001, 
    writeout_dt=Inf, diagnostics_dt=Inf, run_diagnostics=nothing, 
    log=true, restart_from_latest=true,
    show_progress_meter=true, adaptive_dt=true,
    per_step_callback=nothing)

    if restart_from_latest
        t = restart_from!(sim, d, t_end, most_recent_frame(d))
    else
        t = 0.0
    end
    dt = initial_dt

    prog = progress_bar(sim, show_progress_meter, t_end)

    if 0 < diagnostics_dt < Inf
        @assert !isnothing(run_diagnostics)
    elseif isnothing(run_diagnostics)
        @assert diagnostics_dt == Inf
    end
    has_diagnostics = 0 < diagnostics_dt < Inf && !isnothing(run_diagnostics)
    has_writeouts = 0 < writeout_dt < Inf
    has_callback = !isnothing(per_step_callback)

    if has_diagnostics
        t == 0.0 && diagnostics_initial(sim, d, run_diagnostics)
        diagnostics_times = (diagnostics_dt:diagnostics_dt:t_end) ∪ t_end
        next_diagnostic = searchsortedfirst(diagnostics_times, t)
        t_diag = diagnostics_times[next_diagnostic]
    end
    if has_writeouts
        t == 0.0 && writeout_initial(sim, d)
        writeout_times = (writeout_dt:writeout_dt:t_end) ∪ t_end
        next_writeout = searchsortedfirst(writeout_times, t)
        t_writeout = writeout_times[next_writeout]
    end

    while t < t_end
        stepdt = min(dt, t_end-t)
        @timeit "step" success, safety_factor = step!(sim, t, stepdt)

        if !success
            if adaptive_dt
                log && @info "Decreasing dt by factor of $(safety_factor)"
                dt /= safety_factor
                continue
            else
                error("Failed step due to timestep restriction.\nMaximum permissible timestep was $(dt / safety_factor)")
            end
        end

        @assert safety_factor <= 1.0

        t = min(t+stepdt, t_end) # Avoid floating point shenanigans

        show_progress_meter && update_progress!(prog, t, stepdt)
        log && (!show_progress_meter) && should_perform_io(sim) && @show t

        if has_callback
            per_step_callback(sim, t, mksimpath(d))
        end

        if has_diagnostics
            if t >= t_diag
                log && @info "Running diagnostics" t
                diagnostics_soln(sim, t, d, run_diagnostics)
                next_diagnostic = max(next_diagnostic+1, searchsortedfirst(diagnostics_times, t))
                if next_diagnostic <= length(diagnostics_times)
                    t_diag = diagnostics_times[next_diagnostic]
                else
                    t_diag += diagnostics_dt
                end
            end
        end

        if has_writeouts
            if t >= t_writeout
                log && @info "Writing out solution" t
                @timeit "writeout" writeout_solution(sim, t, d)
                next_writeout = max(next_writeout+1, searchsortedfirst(writeout_times, t))
                if next_writeout <= length(writeout_times)
                    t_writeout = writeout_times[next_writeout]
                else
                    t_writeout += writeout_dt
                end
            end
        end
    end
    show_progress_meter && finish!(prog)
end

const PROGRESS_FACTOR = 1e12

progress_bar(sim, show_progress_meter, t_end) = begin
    if should_perform_io(sim) && show_progress_meter
        t_big = Integer(round(big(t_end) * PROGRESS_FACTOR))
        return Progress(t_big)
    else
        return nothing
    end
end

update_progress!(progress, t, dt) = begin
    step = Int(round(big(dt) * PROGRESS_FACTOR))
    tstr = @sprintf("t = %10.4f", t)
    progress.desc = tstr
    if !isnothing(progress)
        next!(progress, step=step)
    end
end

function most_recent_frame(d)
    datafile = joinpath(mksimpath(d), "writeouts.jld2")
    if !isfile(datafile)
        return nothing
    end
    jldopen(datafile, "r") do file
        ks = keys(file)
        ks = filter(startswith("frame_"), ks)
        frs = [parse(Int, split(k, "_")[2]) for k in ks]
        isempty(frs) && return nothing
        return maximum(frs)
    end
end

function restart_from!(sim, d, t_end, fr=most_recent_frame(d))
    datafile = joinpath(mksimpath(d), "writeouts.jld2")
    if isnothing(fr)
        return 0.0
    end
    @info "Restarting from frame_$fr"
    t = jldopen(datafile, "r") do file
        frame = file["frame_$fr"]
        load_from_frame!(sim, frame)
        frame["t"]
    end
    if t >= t_end
        @info "frame_$fr is already at t=$t_end"
        return t
    end
    copy = joinpath(mksimpath(d), "up_to_frame_$fr.jld2")
    should_perform_io(sim) && run(`cp -f $datafile $copy`)
    return t
end

function load_from_frame!(sim, frame) end

end # module PDEHarness
