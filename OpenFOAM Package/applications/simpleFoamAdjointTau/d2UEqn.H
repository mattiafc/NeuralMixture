    // Momentum predictor

    MRF.correctBoundaryVelocity(d2U);

    tmp<fvVectorMatrix> td2UEqn
    (
        
      fvm::div(phi, d2U)
      + MRF.DDt(d2U)
      - fvm :: laplacian(nu, d2U)
      + 2. * (fvc::grad(dU)&dU) + (fvc::grad(U)&d2U)
     ==
        fvOptions(d2U) 
    );
    fvVectorMatrix& d2UEqn = td2UEqn.ref();

    d2UEqn.relax();

    fvOptions.constrain(d2UEqn);

    if (simple.momentumPredictor())
    {
        solve(d2UEqn == -fvc::grad(d2p)  );

        fvOptions.correct(d2U);
    }
