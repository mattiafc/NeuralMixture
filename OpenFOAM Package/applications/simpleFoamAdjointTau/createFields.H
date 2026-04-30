Info<< "Reading transportProperties\n" << endl;

IOdictionary transportProperties
(
    IOobject
    (
        "transportProperties",
        runTime.constant(),
        mesh,
        IOobject::MUST_READ_IF_MODIFIED,
        IOobject::NO_WRITE
    )
);

dimensionedScalar nu
(
    "nu",
    dimViscosity,
    transportProperties
);

dimensionedScalar descent_step
(
    "descent_step",
    transportProperties
);

/*
dimensionedScalar lambda_reg
(
    "lambda_reg",
    transportProperties
);
*/

dimensionedScalar Tf_coef
(
    "Tf_coef",
    transportProperties
);


dimensionedScalar Tinfty
(
    "Tinfty",
    transportProperties
);



Info<< "Reading field p\n" << endl;
volScalarField p
(
    IOobject
    (
        "p",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Reading field U\n" << endl;
volVectorField U
(
    IOobject
    (
        "U",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);


volVectorField Uone
(
    IOobject
    (
        "Uone",
        "0",
        mesh,
        IOobject::MUST_READ,
        IOobject::NO_WRITE
    ),
    mesh
);


Info<< "Reading field U0\n" << endl;
volVectorField U0
(
    IOobject
    (
        "U0",
        "0",
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);




Info<< "Reading field U high fidelity\n" << endl;
volVectorField U_HF
(
    IOobject
    (
        "U_HF",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);


Info<< "Reading Stress Tensor Tau NASA\n" << endl;
volSymmTensorField Tau_NASA
(
    IOobject
    (
        "Tau_NASA",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);


volSymmTensorField TauOld
(
    IOobject
    (
        "TauOld",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh
);


Info<< "Reading field nut \n" << endl;
volScalarField nut
(
    IOobject
    (
        "nut",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);


Info<< "Reading field nut set to one \n" << endl;
volScalarField nutOne
(
    IOobject
    (
        "nutOne",
        "0",
        mesh,
        IOobject::MUST_READ,
        IOobject::NO_WRITE
    ),
    mesh
);


Info<< "Reading field k high fidelity\n" << endl;
volScalarField k_HF
(
    IOobject
    (
        "k_HF",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);


Info<< "Reading Stress Tensor Tau\n" << endl;
volSymmTensorField Tau
(
    IOobject
    (
        "Tau",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);




//#include "createPhi.H"

Info<< "Reading/calculating face flux field phi\n" << endl;

surfaceScalarField phi
(
 IOobject
 (
     "phi",
     runTime.name(),
     mesh,
     IOobject::READ_IF_PRESENT,
     IOobject::AUTO_WRITE
 ),
 fvc::flux(U)
);


label pRefCell = 0;
scalar pRefValue = 0.0;
setRefCell(p, simple.dict(), pRefCell, pRefValue);
mesh.setFluxRequired(p.name());









Info<< "Reading grad J w.r.t Tau\n" << endl;
volSymmTensorField gradJTau
(
    IOobject
    (
        "gradJTau",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);



volSymmTensorField gradJTauZero
(
    IOobject
    (
        "gradJTauZero",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);


/* Adjoint problem */


Info<< "Reading field q\n" << endl;
volScalarField q
(
    IOobject
    (
        "q",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);


Info<< "Reading field xi\n" << endl;
volVectorField xi
(
    IOobject
    (
        "xi",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);


volVectorField xi0
(
    IOobject
    (
        "xi0",
        "0",
        mesh,
        IOobject::MUST_READ,
        IOobject::NO_WRITE
    ),
    mesh
);


/*
Info<< "Calculating field div(U.xi * I) \n" << endl;
volVectorField divUxi
(
    IOobject
    (
        "divUxi",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
    //T(fvc::grad(xi))&U  // right hand side member of the Adjoint equation, taken in explicite time
    //fvc::div((U&xi) * I)  - (T(fvc::grad(U))&xi)
);
*/


Info<< "Reading/calculating face flux field phi(xi)\n" << endl;

surfaceScalarField phixi
(
 IOobject
 (
     "phixi",
     runTime.name(),
     mesh,
     IOobject::READ_IF_PRESENT,
     IOobject::AUTO_WRITE
 ),
 fvc::flux(xi)
);




label qRefCell = 0;
scalar qRefValue = 0.0;
setRefCell(q, simple.dict(), qRefCell, qRefValue);
mesh.setFluxRequired(q.name());





/* Sensibility problem */

Info<< "Reading field dp\n" << endl;
volScalarField dp
(
    IOobject
    (
        "dp",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);


Info<< "Reading field dU\n" << endl;
volVectorField dU
(
    IOobject
    (
        "dU",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Reading field dU0\n" << endl;
volVectorField dU0
(
    IOobject
    (
        "dU0",
        "0",
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::NO_WRITE
    ),
    mesh
);


Info<< "Reading/calculating face flux field phi(dU)\n" << endl;

surfaceScalarField dphi
(
 IOobject
 (
     "dphi",
     runTime.name(),
     mesh,
     IOobject::READ_IF_PRESENT,
     IOobject::AUTO_WRITE
 ),
 fvc::flux(dU)
);




label dpRefCell = 0;
scalar dpRefValue = 0.0;
setRefCell(dp, simple.dict(), dpRefCell, dpRefValue);
mesh.setFluxRequired(dp.name());




/*--------------------------------*- Fields -*----------------------------------*\
|                                                  						         |
 Second order sensitivity equation of N.S
|        																         |
\*------------------------------------------------------------------------------*/

Info<< "Reading field d2p\n" << endl;
volScalarField d2p
(
    IOobject
    (
        "d2p",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);


Info<< "Reading field d2U\n" << endl;
volVectorField d2U
(
    IOobject
    (
        "d2U",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Reading field d2U0\n" << endl;
volVectorField d2U0
(
    IOobject
    (
        "d2U0",
        "0",
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::NO_WRITE
    ),
    mesh
);


Info<< "Reading/calculating face flux field phi(d2U)\n" << endl;

surfaceScalarField d2phi
(
 IOobject
 (
     "d2phi",
     runTime.name(),
     mesh,
     IOobject::READ_IF_PRESENT,
     IOobject::AUTO_WRITE
 ),
 fvc::flux(d2U)
);




label d2pRefCell = 0;
scalar d2pRefValue = 0.0;
setRefCell(d2p, simple.dict(), d2pRefCell, d2pRefValue);
mesh.setFluxRequired(d2p.name());








/*singlePhaseTransportModel laminarTransport(U, phi);


autoPtr<incompressible::turbulenceModel> turbulence
(
    incompressible::turbulenceModel::New(U, phi, laminarTransport)
);
*/
#include "createMRF.H"
#include "createFvOptions.H"
