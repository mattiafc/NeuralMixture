
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




label qRefCell = 0;
scalar qRefValue = 0.0;
setRefCell(q, simple.dict(), qRefCell, qRefValue);
mesh.setFluxRequired(q.name());


/*singlePhaseTransportModel laminarTransport(U, phi);


autoPtr<incompressible::turbulenceModel> turbulence
(
    incompressible::turbulenceModel::New(U, phi, laminarTransport)
);

#include "createMRF.H"
#include "createFvOptions.H"*/
