    // Momentum predictor

    MRF.correctBoundaryVelocity(U);

    tmp<fvVectorMatrix> tUEqn
    (
        fvm::div(phi, U)
      + MRF.DDt(U)
      - fvm :: laplacian(nu, U)
      //+ turbulence->divDevReff(U)
      + fvc::div(Tau)
     ==
        fvOptions(U) 
    );
    fvVectorMatrix& UEqn = tUEqn.ref();

    UEqn.relax();

    fvOptions.constrain(UEqn);

    if (simple.momentumPredictor())
    {
        solve(UEqn == -fvc::grad(p) );

        fvOptions.correct(U);
    }
