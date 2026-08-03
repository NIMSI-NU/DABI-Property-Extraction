function fun_vec = def_objective_vec(in,idx)

fun_vec = @(CreepParams) objective_vec(CreepParams,in,idx);

end