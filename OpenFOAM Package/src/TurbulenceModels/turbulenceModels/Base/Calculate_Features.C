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
	


List<tmp<volScalarField>> eta_list;

dimensionedScalar epsP("epsP",dimensionSet(0,2,-3,0,0,0,0),1e-5);
dimensionedScalar eps7("eps7",dimensionSet(0,1,-2,0,0,0,0),1e-5);
dimensionedScalar epsS("epsS",dimensionSet(0,0,-2,0,0,0,0),1e-5);


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

Info << "\n Calculate Features ... " << nl << endl;

//volScalarField denominator = eta_1_;

Info << "\n eta_1 Normalized Q-criterion" << nl << endl;
//eta_1_ = (sqr(normFrob_Sij) - sqr(normFrob_Oij)) / max ( min ( sqr(normFrob_Sij) + sqr(normFrob_Oij) , -epsS) , epsS);
//eta_1_ = (sqr(normFrob_Sij) - sqr(normFrob_Oij)) /  max(mag(sqr(normFrob_Sij) + sqr(normFrob_Oij)) , epsS);


eta_list.append((sqr(normFrob_Sij) - sqr(normFrob_Oij)) /  max(mag(sqr(normFrob_Sij) + sqr(normFrob_Oij)) , epsS));


Info << "\n eta_2 Turbulence intensity" << nl << endl;
//eta_2_ = ( k_ ) / ( k_ + 0.5 * magSqr(U) );
eta_list.append(( k_ ) / ( k_ + 0.5 * magSqr(U) ));

Info << "\n eta_3 Turbulent Reynolds number " << nl << endl;
//eta_3_ = min(sqrt(k_)* wallDist(this->mesh_).y() /(100.*this->nu()), 1.); // divided by two
eta_list.append(min(sqrt(k_)* wallDist(this->mesh_).y() /(100.*this->nu()), 1.));

Info << "\n eta_4 Pressure gradient along streamline" << nl << endl;
//eta_4_ = (U&gradP) / max ( min ( (U&gradP) + mag(U)*mag(gradP), -epsP) , epsP);

//eta_4_ = (U&gradP) /  max ( mag((U&gradP) + mag(U)*mag(gradP)) , epsP) ;
eta_list.append((U&gradP) /  max ( mag((U&gradP) + mag(U)*mag(gradP)) , epsP));

Info << "\n eta_5 Ratio of turbulent timescale to mean strain timescale" << nl << endl;
//eta_5_ = ( normFrob_Sij * k_ ) / max ( min ( (normFrob_Sij * k_) + epsilon_, -epsP) , epsP);

//eta_5_ = ( normFrob_Sij * k_ ) / max (  mag((normFrob_Sij * k_) + epsilon_) , epsP);
eta_list.append(( normFrob_Sij * k_ ) / max (  mag((normFrob_Sij * k_) + epsilon_) , epsP));

Info << "\n eta_6 Viscosity ratio" << nl << endl;
//eta_6_ =  nut/( nut + this->nu() );
eta_list.append(nut/( nut + this->nu() ));

Info << "\n eta_7 Ratio of pressure normal stresses to normal shear stresses" << nl << endl;
//eta_7_ = mag(gradP) / max ( min ( mag(gradP) + this->rho_ * U_diag_gradU , -eps7) , eps7);

//eta_7_ = mag(gradP) / max ( mag(mag(gradP) + this->rho_ * U_diag_gradU) , eps7);
eta_list.append(mag(gradP) / max ( mag(mag(gradP) + this->rho_ * U_diag_gradU) , eps7));

//Info << "\n eta_8 Vortex stretching" << nl << endl;
//eta_8_ =  mag(wgradU) / (mag(wgradU) + normFrob_Sij2 )  ;

Info << "\n eta_8 Non orthogonality marker between velocity and its gradient. Gorlé &Iaccarino" << nl << endl;
//eta_8_ =  U_UgradU / max ( min ( U_UgradU + (U_gradU_gradUT_U * mag(U)) , -epsP) , epsP);

//eta_8_ =  U_UgradU / max ( mag(U_UgradU + (U_gradU_gradUT_U * mag(U)) ) , epsP);
eta_list.append(U_UgradU / max ( mag(U_UgradU + (U_gradU_gradUT_U * mag(U)) ) , epsP));

Info << "\n eta_9 Ratio of convection to production k" << nl << endl;
//eta_9_ = Ugradk / max ( min ( Ugradk + mag(Rall_ && this->Sij_dim()) , -epsP) , epsP);

//eta_9_ = Ugradk / max ( mag ( Ugradk + mag(Rall_ && this->Sij_dim()) ) , epsP);
eta_list.append(Ugradk / max ( mag ( Ugradk + mag(Rall_ && this->Sij_dim()) ) , epsP));

Info << "\n eta_10 Ratio of total Reynolds stresses to normal Reynolds stresses" << nl << endl;
//eta_10_ = mag(Rall_) / (mag(Rall_) + k_);
eta_list.append(mag(Rall_) / (mag(Rall_) + k_));

Info << "\n eta_11 ratio of Production, Girimaji et al. (2022)" << nl << endl;
//eta_11_ = Pk_  / max ( min ( Pk_  + epsilon_ , -epsP) , epsP);

//eta_11_ = Pk_  / max ( mag ( Pk_  + epsilon_ ) , epsP);
eta_list.append(Pk_  / max ( mag ( Pk_  + epsilon_ ) , epsP));

Info<< "write features ..." << endl;




Info<< "find the nearest orthonormal components of the features (POD) ..." << endl;


double innerProduct;
const int size = 11;
// Create a 2D matrix to represent the correlation matrix
std::vector<std::vector<double>> A(size, std::vector<double>(size));
//std::vector<std::vector<double>> A_org(size, std::vector<double>(size));
// Fill the matrix with inner products
for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
        
        // Euclidian inner product
        innerProduct = 0.;
        forAll(this->mesh_.cells(), celli){innerProduct += eta_list[i]()[celli] * eta_list[j]()[celli];}
        A[i][j] = innerProduct; //fvc::domainIntegrate( eta_list[i]() * eta_list[j]() ).value(); 
        
        // L2 inner product
//        A[i][j] = fvc::domainIntegrate( eta_list[i]() * eta_list[j]() ).value(); 
        
        //A_org[i][j] = A[i][j];
    }
}


// Power iterations to find POD eigenvectors

double lambda = 0.0;
int maxIterations = 400;
// Store all eigenvectors in a matrix
std::vector<std::vector<double>> eigenvectors(size, std::vector<double>(size, 0.0));
std::vector<double> eigenvalues(size);
// Power iteration for each eigenvector
for (int k = 0; k < size; ++k) {
    std::vector<double> v(size, 1.0/sqrt(11.));
    double normV_sqr;
    double error_sqr;
    std::vector<double> Av(size, 0.0);
    std::vector<double> lastV;
	
    for (int iter = 0; iter < maxIterations; ++iter) {
        lastV = v;

        // Multiply A by the vector
        for (size_t i = 0; i < v.size(); ++i) {
            Av[i] = 0.0;
            for (size_t j = 0; j < v.size(); ++j) {
                Av[i] += A[i][j] * v[j];
            }
        }

        for (size_t i = 0; i < v.size(); ++i) { v[i] = Av[i]; }

        // Normalize the vector
        normV_sqr = 0.0;
        for (size_t i = 0; i < v.size(); ++i) { normV_sqr += pow(v[i],2); }

        for (size_t i = 0; i < v.size(); ++i) { v[i] /= sqrt(normV_sqr); }

        // Check for convergence
        error_sqr = 0.0;
        for (size_t i = 0; i < v.size(); ++i) {
            double diff = v[i] - lastV[i];
            error_sqr += pow(diff,2);
        }

        if (sqrt(error_sqr) < 1e-164) {
            std::cout << "Power method converged after " << iter + 1 << " iterations." << std::endl;
            break;
        }
    }

    // Store the eigenvector in the matrix
    for (int i = 0; i < size; ++i) {
        eigenvectors[k][i] = v[i];
    }
    
    
	lambda = 0.0;
	// compute eigenvalue
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            lambda +=  v[i] * A[i][j] * v[j];
        }
    }
	eigenvalues[k] = fabs(lambda);
	
   // Deflate the matrix A
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            A[i][j] -= lambda * v[i] * v[j];
        }
    }
}


/*


// Print eigenvalues
std::cout << "POD Eigenvalues:" << std::endl;
for (int k = 0; k < size; ++k) {
    std::cout << "POD Eigenvalue " << k+1 << ": " << eigenvalues[k] << std::endl;
}



// to run this uncomment the lines for declaration of A_org 


// Reconstruct matrix A using the spectral decomposition
std::vector<std::vector<double>> reconstructed_A(size, std::vector<double>(size, 0.0));
for (int k = 0; k < size; ++k) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            reconstructed_A[i][j] += eigenvalues[k] * eigenvectors[k][i] * eigenvectors[k][j];
        }
    }
}

// Compare reconstructed A with the original A_org
double decomposition_error_sqr = 0. ;
for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
        decomposition_error_sqr = pow(A_org[i][j] - reconstructed_A[i][j], 2);
        
    }
}
std::cout << "Spectral decomposition error " << sqrt(decomposition_error_sqr) << std::endl;


// nearest orthogonal features components

List<tmp<volScalarField>> eta_orth_list;
for (int k = 0; k < size; ++k) {
	tmp<volScalarField> eta_orth_k = 0. * eta_1_;
	for (int j = 0; j < size; ++j) 
	{
	eta_orth_k.ref() += eta_list[j]() * orth_transf[k][j];
	}
	eta_orth_list.append(eta_orth_k);
}
  
  
Info << "orthogonality check" << endl;
for (int i = 0; i < size; ++i) {
	for (int j = 0; j < size; ++j) {
	innerProduct = 0.;
	forAll(this->mesh_.cells(), celli)
			{
				innerProduct += eta_orth_list[i]()[celli] * eta_orth_list[j]()[celli];
			}
			
	Info << i << "\t" << j << "\t" << innerProduct << endl;
	} 
	}


*/



// nearest orthonormal matrix to the features matrix




std::vector<std::vector<double>> orth_transf(size, std::vector<double>(size, 0.0));
// Fill the matrix with some values
for (int k = 0; k < size; ++k) {
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) {
		    orth_transf[i][j] += 1./sqrt(eigenvalues[k]) * eigenvectors[k][i] * eigenvectors[k][j]; 
		}
	}
}

List<tmp<volScalarField>> eta_orth_list;
for (int k = 0; k < size; ++k) {
	tmp<volScalarField> eta_orth_k = 0. * eta_list[k]();
	for (int j = 0; j < size; ++j) 
	{
	eta_orth_k.ref() += eta_list[j]() * orth_transf[k][j];
	}
	eta_orth_list.append(eta_orth_k);
}


eta_1_.ref() = eta_orth_list[0](); eta_1_.correctBoundaryConditions();
eta_2_ .ref()= eta_orth_list[1](); eta_2_.correctBoundaryConditions();
eta_3_.ref() = eta_orth_list[2](); eta_3_.correctBoundaryConditions();
eta_4_.ref() = eta_orth_list[3](); eta_4_.correctBoundaryConditions();
eta_5_.ref() = eta_orth_list[4](); eta_5_.correctBoundaryConditions();
eta_6_.ref() = eta_orth_list[5](); eta_6_.correctBoundaryConditions();
eta_7_.ref() = eta_orth_list[6](); eta_7_.correctBoundaryConditions();
eta_8_.ref()= eta_orth_list[7](); eta_8_.correctBoundaryConditions();
eta_9_.ref()= eta_orth_list[8](); eta_9_.correctBoundaryConditions();
eta_10_.ref() = eta_orth_list[9](); eta_10_.correctBoundaryConditions();
eta_11_.ref() = eta_orth_list[10](); eta_11_.correctBoundaryConditions();


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














