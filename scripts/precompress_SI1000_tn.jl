include("../tndecoder3d.jl")
using .TNDecoder3D
using PyCall
using JLD2
@pyimport stim

cd(dirname(@__FILE__))  # 设置当前工作目录为脚本所在目录
println("Current directory: ", pwd())  # 打印当前工作目录

function gen_circ_tn(d::Int, p::Float64, bd::Int)
    circuit = stim.Circuit.generated("surface_code:rotated_memory_x", distance=d, rounds=d, after_clifford_depolarization=p, before_measure_flip_probability=p, after_reset_flip_probability=p)
    model = circuit.detector_error_model(decompose_errors=false)
    dem = dem_from_stim(model)

    tn3d,expval,open_index = gen_tn_sc3d_circ(dem, bd; do_gauging=false, canonicalness_target=1e-4)

    # do_gauging，压缩维度为2
    pos_checks = [convert(Tuple{Int,Int,Int}, round.((0.5*pos[1], 0.5*pos[2], pos[3]))) for pos in dem.pos_checks]
    xmin,xmax = extrema([pos[1] for pos in pos_checks])
    ymin,ymax = extrema([pos[2] for pos in pos_checks])
    zmin,zmax = extrema([pos[3] for pos in pos_checks])
    println("uncompress expval: ", expval) 
    save_object("./data/circtn"*string(d)*"_uncompress.jld2", (tn3d, expval, open_index))
    
    svd_eps = 10. ^ (-16)
    final_truncation_dim = 4
    if final_truncation_dim != 0
        for x=xmin:xmax for y=ymin:ymax for z=zmin:zmax
            for pos2 in TNDecoder3D.neighbors(tn3d,(x,y,z)) if pos2>(x,y,z)
                expval += TNDecoder3D.identity_SU!(tn3d, (x,y,z), pos2, final_truncation_dim, svd_eps)
                expval += TNDecoder3D.renormalize_tensor!(tn3d, (x,y,z))
                expval += TNDecoder3D.renormalize_tensor!(tn3d, pos2)
                expval += TNDecoder3D.renormalize_bond_matrix!(tn3d, (x,y,z), pos2)
            end end 
        end end end 
    end 
    println("compress expval: ", expval) 
    save_object("./data/circtn"*string(d)*"_compress.jld2", (tn3d, expval, open_index))
end

gen_circ_tn(3, 0.01, 16)