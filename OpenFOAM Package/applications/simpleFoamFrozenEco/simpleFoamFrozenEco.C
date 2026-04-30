/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Application
    simpleFoam

Group
    grpIncompressibleSolvers

Description
    Steady-state solver for incompressible, turbulent flows.

    \heading Solver details
    The solver uses the SIMPLE algorithm to solve the continuity equation:

        \f[
            \div \vec{U} = 0
        \f]

    and momentum equation:

        \f[
            \div \left( \vec{U} \vec{U} \right) - \div \gvec{R}
          = - \grad p + \vec{S}_U
        \f]

    Where:
    \vartable
        \vec{U} | Velocity
        p       | Pressure
        \vec{R} | Stress tensor
        \vec{S}_U | Momentum source
    \endvartable

    \heading Required fields
    \plaintable
        U       | Velocity [m/s]
        p       | Kinematic pressure, p/rho [m2/s2]
        \<turbulence fields\> | As required by user selection
    \endplaintable

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "simpleControl.H"
#include "fvOptions.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    argList::addNote
    (
        "Steady-state solver for incompressible, turbulent flows."
    );

    #include "postProcess.H"

    #include "addCheckCaseOptions.H"
    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createMesh.H"
    #include "createControl.H"
    #include "createFields.H"
    #include "initContinuityErrs.H"

    turbulence->validate();

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nStarting time loop\n" << endl;

    while (simple.loop())
    {
        Info<< "Time = " << runTime.timeName() << nl << endl;
        
        // --- Pressure-velocity SIMPLE corrector
        {	
            #include "gEqn.H"
            #include "qEqn.H"
            

        }
		
		
		//-----------------------------------------------------------------------------
		
		
		U.write();
		p.write();
		
		
		const dimensionedScalar kMin("kMin", dimensionSet(0,2,-2,0,0,0,0), 1e-8); // safe minimum threshold
		
		bijDeltaFrozen_nondim_ = - turbulence->nut() / turbulence->k() * devSymm(fvc::grad(gDelta)); //(turbulence->k() + kMin);
		bijDeltaFrozen_nondim_.correctBoundaryConditions();
		bijDeltaFrozen_nondim_.write();
		
		//-----------------------------------------------------------------------------

//		// Compute true FV-based divergence of gDelta using its flux
//		surfaceScalarField phigD("phigD", fvc::flux(gDelta));
//		volScalarField div_gDelta("div_gDelta", fvc::div(phigD));

//		// Compute L2 norm of the divergence (√∫(∇·gDelta)² dV)
//		scalar incompressibility_L2 = Foam::sqrt
//		(
//			gSum(sqr(div_gDelta) * mesh.V())
//		);

//		// Report the result
//		Info << "||div(gDelta)||_L2 = " << incompressibility_L2 << nl << endl;

//		// Optionally write the divergence field for visualization/debugging
//		div_gDelta.write();
		
		
		// -------------------------------------------------------------------------
		// Build the residual exactly as (lhs – rhs) of the equation you JUST solved
		// -------------------------------------------------------------------------
		
//		volVectorField R_momentum
//		(
//			"R_momentum",
//			// implicit part that went to the matrix:
//			- fvc::laplacian(turbulence->nut(), gDelta)           // same as in eqn
//		    + fvc::grad(q)                                          // RHS term you used in solve()
//		    + fvc::div(phiFrozen, U_Frozen)                       // convective
//			- fvc::div(turbulence->nuEff()*devTwoSymm(fvc::grad(U_Frozen))) // viscous(FROZEN)
//			- fvc::div(turbulence->nut()*T(fvc::grad(gDelta)))    // explicit nut term 
//		);



		// Compute momentum residual
		volVectorField R_momentum("Residual_momentum",
			fvc::div(phiFrozen, U_Frozen)
		  - fvc::div(turbulence->nuEff() * devTwoSymm(fvc::grad(U_Frozen)))
		  + fvc::div(2.0 * turbulence->k() * bijDeltaFrozen_nondim_)
		  + fvc::grad(q)
		);		
		

		// L^2 norm (true FV integral)
		scalar R_L2 = Foam::sqrt
		(
			gSum(magSqr(R_momentum) * mesh.V())
		);


		Info<< "||R_momentum||_L2 = " << R_L2 << nl << endl;

		R_momentum.write();     // for ParaView if you want






//		// Compute momentum residual
//		volVectorField R_momentum("Residual_momentum",
//			fvc::div(phiFrozen, U_Frozen)
//		  - fvc::div(turbulence->nuEff() * devTwoSymm(fvc::grad(U_Frozen)))
//		  + fvc::div(2.0 * turbulence->k() * bijDeltaFrozen_nondim_)
//		  + fvc::grad(q)
//		);

//		scalar R_L2 = mag(R_momentum)().weightedAverage(mesh.V()).value();
//		Info << "||R_momentum||_L2 = " << R_L2 << nl << endl;

//		// Write residual field to file
//		R_momentum.write();

		//-----------------------------------------------------------------------------
		
		
		
//		#include "UEqn.H"
//        #include "pEqn.H"
		
		laminarTransport.correct();
        turbulence->correct();
        
        
        runTime.write();

        runTime.printExecutionTime(Info);
    }

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
