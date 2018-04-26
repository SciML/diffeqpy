function setup(; timeout=5.0 * 60, pollint=0.1)
  timeout = Float64(timeout)
  pollint = Float64(pollint)

  if timeout <= 0
    @eval using DiffEqPy
    return true
  end

  tsk = @schedule @eval using DiffEqPy
  timedwait(() -> istaskdone(tsk), timeout; pollint=pollint)
  return istaskdone(tsk)
end
