PkDelta_ =  bijDelta_ && symm(tgradU());
Pk_ = nut *S2 - PkDelta_;
Rall_ = ((2.0/3.0)*I)*k_ - nut*twoSymm(fvc::grad(U)) + bijDelta_; //this->R(); //


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
	


List<tmp<volScalarField>> eta_list;

dimensionedScalar epsP("epsP",dimensionSet(0,2,-3,0,0,0,0),1);
dimensionedScalar eps7("eps7",dimensionSet(0,1,-2,0,0,0,0),1);
dimensionedScalar epsS("epsS",dimensionSet(0,0,-2,0,0,0,0),1);
dimensionedScalar epsk("epsk",dimensionSet(0,2,-2,0,0,0,0),1);
dimensionedScalar oneP("oneP",dimensionSet(0,2,-3,0,0,0,0),1);
dimensionedScalar epsnu("epsnu",dimensionSet(0,2,-1,0,0,0,0),1);
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
volScalarField d_ = wallDist(this->mesh_).y();

Info << "\n Calculate dimensional Features ... " << nl << endl;

Info << "\n eta_1 Normalized Q-criterion" << nl << endl;
eta_list.append((sqr(normFrob_Oij) - sqr(normFrob_Sij)) / epsS);


Info << "\n eta_2 Turbulence intensity" << nl << endl;
eta_list.append(( k_ ) / ( k_ + 0.5 * magSqr(U) ));

Info << "\n eta_3 Turbulent Reynolds number " << nl << endl;
//eta_list.append(Foam::tanh( sqrt(k_)*wallDist(this->mesh_).y()/(1000.*this->nu())));
eta_list.append(min( (sqrt(k_)* wallDist(this->mesh_).y()) / (100.*this->nu()), 1.));


Info << "\n eta_4 Pressure gradient along streamline" << nl << endl;
eta_list.append( (U&gradP) / epsP);

Info << "\n eta_5 Ratio of turbulent timescale to mean strain timescale" << nl << endl;
eta_list.append(( normFrob_Sij * k_ ) / epsP);

Info << "\n eta_6 Viscosity ratio" << nl << endl;
eta_list.append(nut/( nut + 100. * this->nu() ));

Info << "\n eta_7 Ratio of pressure normal stresses to normal shear stresses" << nl << endl;
eta_list.append(mag(gradP) / eps7);

Info << "\n eta_8 Non orthogonality marker between velocity and its gradient. Gorlé &Iaccarino" << nl << endl;
eta_list.append( U_UgradU / epsP);

Info << "\n eta_9 Ratio of convection to production k" << nl << endl;
eta_list.append(Ugradk / epsP);

Info << "\n eta_10 Ratio of total Reynolds stresses to normal Reynolds stresses" << nl << endl;
eta_list.append(mag(Rall_) / epsk);

Info << "\n eta_11 ratio of Production, Girimaji et al. (2022)" << nl << endl;
eta_list.append(Pk_  / epsP);

Info<< "write features ..." << endl;





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














