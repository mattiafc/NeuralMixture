    // Momentum predictor

    MRF.correctBoundaryVelocity(dU);

    tmp<fvVectorMatrix> tdUEqn
    (
        
      fvm::div(phi, dU)
      + MRF.DDt(dU)
      - fvm :: laplacian(nu, dU)
      + (fvc::grad(U)&dU) - fvc::div(gradJTau) 
     ==
        fvOptions(dU) 
    );
    fvVectorMatrix& dUEqn = tdUEqn.ref();

    dUEqn.relax();

    fvOptions.constrain(dUEqn);

    if (simple.momentumPredictor())
    {
        solve(dUEqn == -fvc::grad(dp)  );

        fvOptions.correct(dU);
    }
