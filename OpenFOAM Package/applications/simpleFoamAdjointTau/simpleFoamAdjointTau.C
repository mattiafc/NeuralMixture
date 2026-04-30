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
    #include "initContinuityErrs.H"
	#include "createFields.H"
	
	
    //turbulence->validate();
	int iter = 0;
	double Jvalue;
	double dJstep;
	double d2Jstep;
	double norm_nablaJ_wrt_step;
	double eta_desc = 0.;
	double lambda_reg = 1e-3;
	double dJOnd2J;
	double Tlim;
	double JvalueOld;
	OFstream outFileJ("J.txt");
	OFstream outFileStepSize("StepSize.txt");
	
	
	// initialize Tau
	
	//Tau = Tau_NASA-(2./3.)*k_HF*I;
	Tau = - 2. * nut * dev(symm(fvc::grad(U0)));
	while(true)
	{
		// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
		iter++;
		Info<< "\nStarting time loop - Primal problem\n" << endl;
		
		runTime.setTime(runTime.startTime().value(), 0);
		U = U0;
		while (simple.loop())
		{
		    Info<< "Forward Time = " << runTime.timeName() << nl << endl;
			
		    // --- Pressure-velocity SIMPLE corrector
		    {
		        #include "UEqn.H"
		        #include "pEqn.H"
		    }

		    runTime.write();
		    runTime.printExecutionTime(Info);
		    
		   
		}

		Info<< "End of primal problem\n" << endl;
		//std::cin.ignore();
		// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
		
		Info << "Starting backward time loop - Adjoint problem\n" << endl;
		
		xi = - (U - U_HF) * Tf_coef ;
		xi.write();
		
		//xi = xi0;
		//xi.write();
			
		//Calculate the Functional value at the final time
		//Jvalue = fvc::domainIntegrate((U - U_HF)&(U - U_HF)).value();
		//Info << " J(.) = " << Jvalue << endl;
		//outFileJ << iter << "\t" << Jvalue << endl;
		
		// Adjoint time loop (backward in time)
		runTime.setDeltaT(-runTime.deltaT()); // // Reverse the time step Dt to -Dt
		
		gradJTau = gradJTauZero;
		while (simple.loop())
		{
			Info << "Backward Time = " << runTime.timeName() << nl << endl;
			
			if (runTime.time() > Tinfty ) { Tlim = 1.;}
			else {Tlim = 0.;}
			// --- Adjoint Pressure-velocity SIMPLE corrector
		    {   
		        #include "xiEqn.H"
		        #include "qEqn.H"   
		    }
			gradJTau.ref() = nutOne * - dev(symm(fvc::grad(xi))) + lambda_reg * Tau ;
			//gradJTau.ref() += lambda_reg * Tau ;
			gradJTau.correctBoundaryConditions();
			Info << "update Tau\n" << endl;
			Tau.ref() -= descent_step * gradJTau;
			Tau.correctBoundaryConditions();
		
			//runTime.write();
			//runTime.printExecutionTime(Info);
			
			if (runTime.time().value() == runTime.startTime().value())
			{
			break;
			}
			
		}
		// Add regularization term
		
		Info << "End of adjoint problem\n" << endl;
		
		runTime.setDeltaT( - runTime.deltaT());// Reverse the time step -Dt to Dt
		
		
		
		
		Tau.write();
		
		
		//std::cin.ignore();
		
		
		
		// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
		
		
		// Optimal descent
		Info << "Optimal descent start\n" << endl;
		norm_nablaJ_wrt_step = 1.;	
		//eta_desc = 0.;
		TauOld = Tau;
		//Jvalue = 1e5; 
		while (norm_nablaJ_wrt_step>1e-3)
		   {	
				// set time to initial time
				runTime.setTime(runTime.startTime().value(), 0);
				
				Info << "update Tau\n" << endl;
				Tau = TauOld - eta_desc * gradJTau;
				Tau.correctBoundaryConditions();
				
				
				
				U = U0;
				dU = dU0;
				d2U = d2U0;
				//Jvalue = 0.;
				//dJstep = 0.;
				//d2Jstep	= 0.;
				while (simple.loop())
				{
					Info<< "Sensitivities - Time = " << runTime.timeName() << nl << endl;

					// --- Pressure-velocity SIMPLE corrector
					{
						#include "UEqn.H"
						#include "pEqn.H"
					}
					{	
						#include "dUEqn.H"
			  			#include "dpEqn.H"
			  		}
			  		{	
			  			#include "d2UEqn.H"
			  			#include "d2pEqn.H"
					}
					//runTime.write();
					//runTime.printExecutionTime(Info);
					//if (runTime.time() > Tinfty ) 
					if (runTime.time().value() == runTime.endTime().value())
					{ 
					//JvalueOld = Jvalue;
					Jvalue = 0.5 * fvc::domainIntegrate((U - U_HF)&(U - U_HF)).value() + 0.5 * lambda_reg * fvc::domainIntegrate(Tau&&Tau).value();
					//if (Jvalue > JvalueOld) {break;}
					dJstep = fvc::domainIntegrate((U-U_HF)&dU).value() + lambda_reg * fvc::domainIntegrate(-gradJTau&&Tau).value();
					d2Jstep = fvc::domainIntegrate(dU&dU).value() + fvc::domainIntegrate((U-U_HF)&d2U).value() + lambda_reg * fvc::domainIntegrate(-gradJTau&&-gradJTau).value();
					}
				}
				
				
				dJOnd2J =  dJstep / d2Jstep ;

				eta_desc -= dJOnd2J;
				
				norm_nablaJ_wrt_step = fabs(dJstep);
				
				//Info << "norm_nablaJ_wrt_step = " << norm_nablaJ_wrt_step << "\teta_desc = " << eta_desc << endl;
				outFileStepSize << "J = " << Jvalue << "\t dJ = "  << dJstep << "\t d2J = " << d2Jstep << "\t eta_desc = " << eta_desc << endl;
			}
		outFileStepSize << "--------------------------------------------------------------------------------" << endl;
		Info << "Optimal Step Found\n" << endl;
		
		
		
		
     }
     

    return 0;
}


// ************************************************************************* //
