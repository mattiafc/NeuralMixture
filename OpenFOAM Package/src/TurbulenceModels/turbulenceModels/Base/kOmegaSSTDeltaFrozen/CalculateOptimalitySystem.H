Info << "Writing minRes projection files at the last time step\n";
//volScalarField Residual_(ReskOmega_);
//volScalarField Residual_ = ReskOmega_;
const int lambdaMax = 3; // 2D case, the number of terms is 3, 3D case it is equal to 10
int h_max = this->monomialExponents().size();
double integralValue;

// Create file for the lhs/rhs of the min Problem to find TwokbDelta and bR
OFstream outFileLhsDelta("./ThetaCoefs/LhsMatrixDelta.txt");
OFstream outFileRhsDelta("./ThetaCoefs/RhsVectorDelta.txt");
OFstream outFileLhsR("./ThetaCoefs/LhsMatrixR.txt");
OFstream outFileRhsR("./ThetaCoefs/RhsVectorR.txt");
if (!outFileLhsDelta.good() || !outFileRhsDelta.good() || !outFileLhsR.good() || !outFileRhsR.good()) {
FatalErrorInFunction << "Failed to open one of the outfiles lhsDelta || rhsDelta || lhsR || rhsR !" << exit(FatalError);}


List<tmp<volScalarField>> prodIvExponents = this->prodIvExponents();
List<tmp<volSymmTensorField>> TList = this->TList();
List<tmp<volScalarField>> TListGradU = this->TListGradU(TList, tgradU);

for (int li_ = 0; li_ < lambdaMax; ++li_) 
	{
	for (int hi_ = 0; hi_ < h_max; ++hi_)
		{
		Info << "row " << h_max * li_ + hi_ << "  ..." << endl;
		// The parenthesis make the integral workout the difference between types : DimensionedField and GeometricField
		integralValue = fvc::domainIntegrate( prodIvExponents[hi_]() * (TList[li_]() && bijDeltaFrozen_nondim) ).value();
		outFileRhsDelta << h_max * li_ + hi_ << "\t" << integralValue << endl;
		//Info << "i = " << h_max * li_ + hi_ << integralValue << endl;
		
		integralValue = fvc::domainIntegrate( prodIvExponents[hi_]() * (TListGradU[li_]() * ReskOmegaOn2k_) ).value();
		outFileRhsR << h_max * li_ + hi_ << "\t" << integralValue << endl;
		//Info << "i = " << h_max * li_ + hi_ << integralValue << endl;
		// -------------------------------------------------------------------------------------------------------------------
			for (int lj_ = 0; lj_ < lambdaMax; ++lj_) 
			{ 
				for (int hj_ = 0; hj_ < h_max; ++hj_)
				{
				// lhs matrix
				if (h_max * lj_ + hj_ >= h_max * li_ + hi_)
				{
				
				integralValue = fvc::domainIntegrate(
													 (prodIvExponents[hi_]() * prodIvExponents[hj_]()) 
													 * (TList[li_]()&&TList[lj_]())
													).value();
				
				outFileLhsDelta << h_max * li_ + hi_ << "\t" << h_max * lj_ + hj_ << "\t" << integralValue << endl;
				//Info << "i = " << h_max * li_ + hi_ << " \t j = " << h_max * lj_ + hj_ << integralValue << endl;
				
				integralValue = fvc::domainIntegrate(
													(prodIvExponents[hi_]() * TListGradU[li_]())
													*
													(prodIvExponents[hj_]() * TListGradU[lj_]())
												   ).value();
				
				outFileLhsR << h_max * li_ + hi_ << "\t" << h_max * lj_ + hj_ << "\t" << integralValue << endl;
				
				//Info << "i = " << h_max * li_ + hi_ << " \t j = " << h_max * lj_ + hj_ << integralValue << endl;
				}
				}
			}
		// -------------------------------------------------------------------------------------------------------------------
		}
	}

// Close the file
Info << "Files  lhsDelta, rhsDelta, lhsR, rhsR  have been written." << endl;
