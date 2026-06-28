# Paper assets v32 multi-seed routing

## Main update

The learned routing claim is now seed-aware.
The pre-specified ridge expected-utility router is stably positive on H=48 for all tested compute costs.
It is not stably positive on H=36 or H=72.

## Recommended claim

A learned multi-capacity router can improve over the best fixed capacity on specific hard OOD regimes.
In the current benchmark, this effect is robust on the H=48 constrained-geometry shift.
The hardest H=72 shift retains large oracle headroom, indicating that better routing signals are needed.

## Key numbers

H=48 ridge expected-utility gains:
- λ=0.00: +0.0265, CI [0.0083, 0.0448].
- λ=0.05: +0.0350, CI [0.0173, 0.0527].
- λ=0.10: +0.0439, CI [0.0120, 0.0733].
- λ=0.20: +0.0460, CI [0.0152, 0.0749].
- λ=0.30: +0.0467, CI [0.0180, 0.0726].

H=72:
- learned gains are not stable;
- oracle headroom stays around +0.11 to +0.12;
- this should be framed as an unsolved routing-signal problem.

## Claim boundary

Report the pre-specified ridge router as the clean learned-routing result.
Best-of-method tables can be shown in appendix or diagnostics, but the main text should avoid cherry-picking routers.
