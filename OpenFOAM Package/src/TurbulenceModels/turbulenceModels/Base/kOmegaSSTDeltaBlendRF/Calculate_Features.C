PkDelta_ =  bijDelta_ && symm(tgradU());
Pk_ = nut *S2 - PkDelta_;
Rall_ = ((2.0/3.0)*I)*k_ - nut*twoSymm(fvc::grad(U)) + bijDelta_;


volScalarField p_
	(
		IOobject
		(
		    "p",
		    this->runTime_.timeName(),
		    this->mesh_,
		    IOobject::MUST_READ,
		    IOobject::NO_WRITE
		),
		this->mesh_
	);
	
dimensionedScalar gradPmin("gradPmin",dimensionSet(0,2,-3,0,0,0,0),1e-0);//1e-4);//1e-15);
dimensionedScalar min12("min12",dimensionSet(0,1,-2,0,0,0,0),1e-0);//1e-3);
dimensionedScalar cstS("cstS",dimensionSet(0,0,-2,0,0,0,0),1e-6);//1e-3);

//tmp<volScalarField> d = wallDist(mesh_).y();

//volVectorField gradP = fvc::grad(p_);

volVectorField gradP = fvc::grad(p_);



volVectorField UgradU = fvc::div(phi_, U);//U & tgradU; //

volTensorField gradU = fvc::grad(U);

U_diag_gradU = U.component(vector::X) * gradU.component(tensor::XX)
			 + U.component(vector::Y) * gradU.component(tensor::YY)
			 + U.component(vector::Z) * gradU.component(tensor::ZZ);


volScalarField U_UgradU =  U&UgradU;
volScalarField U_gradU_gradUT_U =  mag(T(tgradU())&U);//sqrt((U&(tgradU()&T(tgradU())))&U);
volScalarField normFrob_Sij2 = tr( this->Sij_dim() & this->Sij_dim() ) ;
volScalarField normFrob_Sij = sqrt( tr( this->Sij_dim() & this->Sij_dim() ) );
volScalarField normFrob_Oij = sqrt(tr(T(this->Oij_dim())&this->Oij_dim()));
//volVectorField wgradU = vorticity_ & tgradU;
volScalarField Ugradk = U & fvc::grad(k_);
Info << "\n Calculate Features " << nl << endl;

Info << "\n eta_1 Normalized Q-criterion" << nl << endl;
eta_1_ = (sqr(normFrob_Sij) - sqr(normFrob_Oij)) / (sqr(normFrob_Sij) + sqr(normFrob_Oij) + scalar(1)*cstS ) ;
//eta_1_ = max ( min ( eta_1_, 1.) , -1.);
//eta_1_ = Foam::tanh(eta_1_);

Info << "\n eta_2 Turbulence intensity" << nl << endl;
eta_2_ = ( k_ ) / ( k_ + 0.5 * magSqr(U) );
//eta_2_ = max ( min ( eta_2_, 1.) , -1.);
//eta_2_ = Foam::tanh(eta_2_);

Info << "\n eta_3 Turbulent Reynolds number " << nl << endl;
//eta_3_ = Foam::tanh( sqrt(k_)* wallDist(this->mesh_).y() /(50.*this->nu())) ;
eta_3_ = min(sqrt(k_)* wallDist(this->mesh_).y() /(50.*this->nu()), 2.); // divided by 2
//eta_3_ = max ( min ( eta_3_, 1.) , -1.);
//eta_3_ = Foam::tanh(eta_3_);

Info << "\n eta_4 Pressure gradient along streamline" << nl << endl;
//eta_4_ = scalar(1) -  mag(U)*mag(gradP) / (mag(U)*mag(gradP) + mag(U & gradP) + scalar(1)*gradPmin);
eta_4_ = (U&gradP) / ((U&gradP) + mag(U)*mag(gradP) + scalar(1)*gradPmin);
//eta_4_ = max ( min ( eta_4_, 1.) , -1.);
//eta_4_ = Foam::tanh(eta_4_);

Info << "\n eta_5 Ratio of turbulent timescale to mean strain timescale" << nl << endl;
eta_5_ = ( normFrob_Sij * k_ ) / ( (normFrob_Sij * k_) + epsilon_+ scalar(1)*gradPmin) ;// + scalar(1)*gradPmin );
//eta_5_ = max ( min ( eta_5_, 1.) , -1.);
//eta_5_ = Foam::tanh(eta_5_);

Info << "\n eta_6 Viscosity ratio" << nl << endl;
eta_6_ =  nut/( nut + this->nu() );
//eta_6_ = max ( min ( eta_6_, 1.) , -1.);
//eta_6_ = Foam::tanh(eta_6_);

Info << "\n eta_7 Ratio of pressure normal stresses to normal shear stresses" << nl << endl;
eta_7_ = mag(gradP)/ ( mag(gradP) + this->rho_ * U_diag_gradU + scalar(1)*min12);
//eta_7_ = max ( min ( eta_7_, 1.) , -1.);
//eta_7_ = Foam::tanh(eta_7_);

//Info << "\n eta_8 Vortex stretching" << nl << endl;
//eta_8_ =  mag(wgradU) / (mag(wgradU) + normFrob_Sij2 )  ;

Info << "\n eta_8 Non orthogonality marker between velocity and its gradient. Gorlé &Iaccarino" << nl << endl;
eta_8_ =  U_UgradU / (U_UgradU + (U_gradU_gradUT_U * mag(U)) + scalar(1)*gradPmin);
//eta_8_ = max ( min ( eta_8_, 1.) , -1.);
//eta_8_ = Foam::tanh(eta_8_);

Info << "\n eta_9 Ratio of convection to production k" << nl << endl;
eta_9_ = Ugradk / (Ugradk + mag(Rall_ && this->Sij_dim()) + scalar(1)*gradPmin);
//eta_9_ = max ( min ( eta_9_, 1.) , -1.);
//eta_9_ = Foam::tanh(eta_9_);

Info << "\n eta_10 Ratio of total Reynolds stresses to normal Reynolds stresses" << nl << endl;
eta_10_ = mag(Rall_) / (mag(Rall_) + k_);
//eta_10_ = max ( min ( eta_10_, 1.) , -1.);
//eta_10_ = Foam::tanh(eta_10_);

Info << "\n eta_11 ratio of Production, Girimaji et al. (2022)" << nl << endl;
eta_11_ = Pk_  / ( Pk_  + epsilon_ + gradPmin);
//eta_11_ = max ( min ( eta_11_, 1.) , -1.);
//eta_11_ = Foam::tanh(eta_11_);


Info<< "write features ..." << endl;


eta_1_.write();
eta_2_.write();
eta_3_.write();
eta_4_.write();
eta_5_.write();
eta_6_.write();
eta_7_.write();
eta_8_.write();
eta_9_.write();
eta_10_.write();
eta_11_.write();


































//Info << "\n eta_1 Normalized Q-criterion" << nl << endl;
//eta_1_ = (sqr(normFrob_Sij) - sqr(normFrob_Oij)) / (sqr(normFrob_Sij) + sqr(normFrob_Oij) + scalar(1)*cstS) ;

//Info << "\n eta_2 Turbulence intensity" << nl << endl;
//eta_2_ = ( k_ ) / ( k_ + 0.5 * magSqr(U) );


//Info << "\n eta_3 Turbulent Reynolds number " << nl << endl;
////eta_3_ = Foam::tanh( sqrt(k_)* wallDist(this->mesh_).y() /(50.*this->nu())) ;
//eta_3_ = min(sqrt(k_)* wallDist(this->mesh_).y() /(100.*this->nu()), 1.); // divided by 2


//Info << "\n eta_4 Pressure gradient along streamline" << nl << endl;
////eta_4_ = scalar(1) -  mag(U)*mag(gradP) / (mag(U)*mag(gradP) + mag(U & gradP) + scalar(1)*gradPmin);
//eta_4_ = (U&gradP) / ((U&gradP) + mag(U)*mag(gradP) + scalar(1)*gradPmin);


//Info << "\n eta_5 Ratio of turbulent timescale to mean strain timescale" << nl << endl;
//eta_5_ = ( normFrob_Sij * k_ ) / ( (normFrob_Sij * k_) + epsilon_+ scalar(1)*gradPmin) ;// + scalar(1)*gradPmin );


//Info << "\n eta_6 Viscosity ratio" << nl << endl;
//eta_6_ =  nut/( nut + this->nu() );


//Info << "\n eta_7 Ratio of pressure normal stresses to normal shear stresses" << nl << endl;
//eta_7_ = mag(gradP)/ ( mag(gradP) + this->rho_ * U_diag_gradU + scalar(1)*min12);


//Info << "\n eta_8 Non orthogonality marker between velocity and its gradient. Gorlé &Iaccarino" << nl << endl;
//eta_8_ =  U_UgradU / (U_UgradU + (U_gradU_gradUT_U * mag(U)) + scalar(1)*gradPmin);


//Info << "\n eta_9 Ratio of convection to production k" << nl << endl;
//eta_9_ = Ugradk / (Ugradk + mag(Rall_ && this->Sij_dim()) + scalar(1)*gradPmin);

//Info << "\n eta_10 Ratio of total Reynolds stresses to normal Reynolds stresses" << nl << endl;
//eta_10_ = mag(Rall_) / (mag(Rall_) + k_);

//Info << "\n eta_11 ratio of Production, Girimaji et al. (2022)" << nl << endl;
//eta_11_ = Pk_  / ( Pk_  + epsilon_ + gradPmin);

