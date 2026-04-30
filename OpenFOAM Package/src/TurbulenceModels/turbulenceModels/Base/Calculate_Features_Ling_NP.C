PkDelta_ =  bijDelta_ && symm(tgradU());
Pk_ = nut *S2 - PkDelta_;
Rall_ = ((2.0/3.0)*I)*k_ - nut*twoSymm(fvc::grad(U)) + bijDelta_; //this->R(); //

//Info << "\n STEP0... " << nl << endl;
// volScalarField p_
// 	(
// 		IOobject
// 		(
// 		    "p",
// 		    this->runTime_.timeName(),
// 		    this->mesh_,
// 		    IOobject::MUST_READ,
// 		    IOobject::NO_WRITE
// 		),
// 		this->mesh_
// 	);


List<tmp<volScalarField>> eta_list;

dimensionedScalar rhoMin1
		(
			"rhoMin1",  // Name
			dimDensity, // Ensures correct units (kg/m³)
			1e-6       // Universal small value
		);

//dimensionedScalar epsP
//(
//    "epsP",
//    (this->rho_.dimensions() == dimDensity)
//        ? dimensionSet(1, -1, -3, 0, 0, 0, 0)  // Compressible case (UgradU dimensions)
//        : dimensionSet(0, 2, -3, 0, 0, 0, 0), // Incompressible case (epsP dimensions)
//    1e-5 // Value remains constant
//);

dimensionedScalar epsP("epsP",dimensionSet(0,2,-3,0,0,0,0),1e-5);
dimensionedScalar eps7("eps7",dimensionSet(0,1,-2,0,0,0,0),1e-5);
dimensionedScalar epsS("epsS",dimensionSet(0,0,-2,0,0,0,0),1e-5);

dimensionedScalar oneP("oneP",dimensionSet(0,2,-3,0,0,0,0),1);

//tmp<volScalarField> d = wallDist(mesh_).y();

//volVectorField gradP = fvc::grad(p_);

Info << "\n Get p" << nl << endl;

volVectorField gradP = fvc::grad(pTurb_);




volVectorField UgradU =
    (this->rho_.dimensions() == dimDensity)
        ? (1.0 / (this->rho_ + rhoMin1)) * fvc::div(this->phi_, U) - fvc::div(U) * U
        : fvc::div(this->phi_, U);


//volVectorField UgradU = (1.0 / (this->rho_ + rhoMin1)) * fvc::div(this->phi_, this->rho_ * U);

//volVectorField UgradU = fvc::div(this->phi_, U) - fvc::div(this->phi_, U);//tgradU & U;

volTensorField gradU = fvc::grad(U);

U_diag_gradU = U.component(vector::X) * gradU.component(tensor::XX)
			 + U.component(vector::Y) * gradU.component(tensor::YY)
			 + U.component(vector::Z) * gradU.component(tensor::ZZ);


volScalarField U_UgradU =  U&UgradU;
volScalarField U_gradU_gradUT_U =  mag(UgradU) * mag(U);//sqrt((U&(tgradU()&T(tgradU())))&U);
volScalarField normFrob_Sij2 = tr( this->Sij_dim() & this->Sij_dim() ) ;
volScalarField normFrob_Sij = sqrt( mag(tr( this->Sij_dim() & this->Sij_dim() ) ) );
volScalarField normFrob_Oij = sqrt( mag(tr(T(this->Oij_dim())&this->Oij_dim() ) ) );
//volVectorField wgradU = vorticity_ & tgradU;
volScalarField Ugradk = U & fvc::grad(k_);

Info << "\n Calculate Features ... " << nl << endl;

//Info << "\n eta_1 Normalized Q-criterion" << nl << endl;
eta_list.append((sqr(normFrob_Oij) - sqr(normFrob_Sij)) /  (sqr(normFrob_Sij) + sqr(normFrob_Oij) + epsS));

//Info << "\n eta_2 Turbulence intensity" << nl << endl;
eta_list.append(( k_ ) / ( k_ + 0.5 * magSqr(U) ));

//Info << "\n eta_3 Turbulent Reynolds number " << nl << endl;
//eta_list.append(Foam::tanh( sqrt(k_)*wallDist(this->mesh_).y()/(1000.*this->nu())));
eta_list.append(min(sqrt(k_)* wallDist(this->mesh_).y() /(100.*this->nu()), 1.));

//Info << "\n eta_4 Pressure gradient along streamline" << nl << endl;
//eta_list.append( (U&gradP) /  ( mag(U&gradP) + mag(U)*mag(gradP) + epsP));

eta_list.append( ((U & gradP) / this->rho_) / 
               ( mag((U & gradP) / this->rho_) + mag(U)*mag(gradP/this->rho_) + epsP) );


//Info << "\n eta_5 Ratio of turbulent timescale to mean strain timescale" << nl << endl;
eta_list.append(( normFrob_Sij * k_ ) / (  (normFrob_Sij * k_) + epsilon_));

//Info << "\n eta_6 Viscosity ratio" << nl << endl;
eta_list.append(nut/( nut + 100. * this->nu() ));

//Info << "\n eta_7 Ratio of pressure normal stresses to normal shear stresses" << nl << endl;
//eta_list.append(mag(gradP) / ( mag(gradP) + this->rho_ * mag(U_diag_gradU) + eps7));

eta_list.append(mag(gradP / this->rho_) / 
               ( mag(gradP / this->rho_) + mag(U_diag_gradU) + eps7) );


//Info << "\n eta_8 Non orthogonality marker between velocity and its gradient. Gorlé &Iaccarino" << nl << endl;
eta_list.append( U_UgradU / ( mag(U_UgradU) + U_gradU_gradUT_U + epsP) ); //

//eta_list.append( (U_UgradU / this->rho_) / 
//                 ( mag(U_UgradU / this->rho_) + mag(U_gradU_gradUT_U) * mag(U) + epsP) );


//Info << "\n eta_9 Ratio of convection to production k" << nl << endl;
eta_list.append(Ugradk / ( mag(Ugradk) + mag(Rall_ && this->Sij_dim())  + epsP));

//Info << "\n eta_10 Ratio of total Reynolds stresses to normal Reynolds stresses" << nl << endl;
eta_list.append(mag(Rall_) / (mag(Rall_) + k_));

//Info << "\n eta_11 ratio of Production, Girimaji et al. (2022)" << nl << endl;
eta_list.append(Pk_  / ( mag(Pk_)  + epsilon_  + epsP));


//Info << "\n eta_12 Ratio of turbulent timescale to mean rotation timescale" << nl << endl;
eta_list.append(( normFrob_Oij * k_ ) / (  (normFrob_Oij * k_) + epsilon_));


//Info<< "write features ..." << endl;





eta_1_.ref() = eta_list[0](); eta_1_.correctBoundaryConditions();
eta_2_ .ref()= eta_list[1](); eta_2_.correctBoundaryConditions();
eta_3_.ref() = eta_list[2](); eta_3_.correctBoundaryConditions();
eta_4_.ref() = eta_list[3](); eta_4_.correctBoundaryConditions();
eta_5_.ref() = eta_list[4](); eta_5_.correctBoundaryConditions();
eta_6_.ref() = eta_list[5](); eta_6_.correctBoundaryConditions();
eta_7_.ref() = eta_list[6](); eta_7_.correctBoundaryConditions();
eta_8_.ref()= eta_list[7](); eta_8_.correctBoundaryConditions();
eta_9_.ref()= eta_list[8](); eta_9_.correctBoundaryConditions();
eta_10_.ref() = eta_list[9](); eta_10_.correctBoundaryConditions();
eta_11_.ref() = eta_list[10](); eta_11_.correctBoundaryConditions();
eta_12_.ref() = eta_list[11](); eta_12_.correctBoundaryConditions();














