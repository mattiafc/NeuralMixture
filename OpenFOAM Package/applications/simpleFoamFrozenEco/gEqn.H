

tmp<fvVectorMatrix> tgDeltaEqn
(
   - fvm::laplacian(turbulence->nut(), gDelta)               // Implicit symmetric gradient part
/*    -  fvm::div(turbulence->nut() * (fvc::grad(gDelta) + T(fvc::grad(gDelta))) )               // Implicit symmetric gradient part*/
 ==
   - fvc::div(phiFrozen, U_Frozen)                                              // Convective
   + fvc::div(turbulence->nuEff() * devTwoSymm(fvc::grad(U_Frozen)))       // Viscous stress
   + fvc::div(turbulence->nut() * T(fvc::grad(gDelta)))     // Explicit correction
   + fvOptions(gDelta)                                              // Source terms
);


fvVectorMatrix& gDeltaEqn = tgDeltaEqn.ref();



gDeltaEqn.relax();
fvOptions.constrain(gDeltaEqn);

if (simple.momentumPredictor())
{
    solve(gDeltaEqn == -fvc::grad(q));
    fvOptions.correct(gDelta);
}



