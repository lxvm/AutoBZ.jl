# Adaptive integration

## Routines

```@docs
AutoBZ.AutoBZCore.iterated_integration
AutoBZ.AutoBZCore.alloc_segbufs
AutoBZ.AutoBZCore.thunk
AutoBZ.AutoBZCore.ThunkIntegrand
```

## Customization

The methods below can be extended/customized to integrand types

```@docs
AutoBZ.AutoBZCore.iterated_tol_update
AutoBZ.AutoBZCore.iterated_pre_eval
AutoBZ.AutoBZCore.iterated_segs
AutoBZ.AutoBZCore.infer_f
```