// @HEADER
// ***********************************************************************
//
//           Panzer: A partial differential equation assembly
//       engine for strongly coupled complex multiphysics systems
//                 Copyright (2011) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Roger P. Pawlowski (rppawlo@sandia.gov) and
// Eric C. Cyr (eccyr@sandia.gov)
// ***********************************************************************
// @HEADER

#ifndef PANZER_PARAMETER_LIBRARY_UTILITIES_IMPL_HPP
#define PANZER_PARAMETER_LIBRARY_UTILITIES_IMPL_HPP

namespace panzer {

  template<typename EvaluationType>
  Teuchos::RCP<panzer::ScalarParameterEntry<EvaluationType> >
  createAndRegisterScalarParameter(const std::string name, panzer::ParamLib& pl)
  {
    if (!pl.isParameter(name))
      pl.addParameterFamily(name,true,false);
    
    Teuchos::RCP<panzer::ScalarParameterEntry<EvaluationType> > entry;
    
    if (pl.isParameterForType<EvaluationType>(name)) {
      Teuchos::RCP<Sacado::ScalarParameterEntry<EvaluationType,panzer::SacadoScalarTypeTraits> > sacado_entry =
	pl.getEntry<EvaluationType>(name);
      entry = Teuchos::rcp_dynamic_cast<panzer::ScalarParameterEntry<EvaluationType> >(sacado_entry);
    }
    else {
      entry = Teuchos::rcp(new panzer::ScalarParameterEntry<EvaluationType>);
      entry->setValue(NAN);
      pl.addEntry<EvaluationType>(name,entry);
    }

    return entry;
  }
  
  /*template<typename EvaluationType>
  std::shared_ptr<TianXin::GeneralFunctor<EvaluationType> >
  createAndRegisterFunctor(const Teuchos::ParameterList& pl, panzer::FunctorLib& pfl)
  {
    //if (pl.find(name) != pl.end() )
    //  pl.addParameterFamily(name,true,false);
    
    std::string pname = pl.name();
	
	if (pfl.find(pname) != pfl.end() ) 
		return pfl[pname];
	else {
		auto& value_type = pl.get<std::string>("Value Type");
		if( value_type.find("Parameter") != std::string::npos )  // Global parameter
			return nullptr;
		
		if( value_type == "Constant" ) {
			std::shared_ptr<TianXin::GeneralFunctor<EvaluationType>> pC(new TianXin::ConstantFunctor<EvaluationType>(pl));
			pfl.insert( std::make_pair(pname, pC) );
			return pC;
		}
	}
  }*/
  
  template<typename EvaluationType>
  void 
  createAndRegisterFunctor(const Teuchos::ParameterList& ppl, std::unordered_map<std::string, panzer::FunctorLib>& pfls)
  {
	typedef Teuchos::ParameterList::ConstIterator pl_iter;
	for (pl_iter input = ppl.begin(); input != ppl.end(); ++input) {
		TEUCHOS_TEST_FOR_EXCEPTION( !(input->second.isList()), std::logic_error,
                            "All entries in the material block must be a sublist!" );

        auto const& pl = ppl.sublist(input->first);
		panzer::FunctorLib pfl;
		auto const& name = pl.name();
		std::size_t found = name.find_last_of("->");
	    std::string nname = name.substr(found+1);
		for(auto it = pl.begin(); it != pl.end(); ++it) {
			try {
				const auto& pll = pl.sublist(it->first);
				Teuchos::ParameterList pp(pll);
				std::string ppname = pp.name();
				std::size_t found = ppname.find_last_of("->");
			    std::string pname = ppname.substr(found+1);
				auto& value_type = pp.get<std::string>("Value Type","Constant");
				if( value_type.find("Parameter") != std::string::npos ) continue;
				pfl[pname] = TianXin::GeneralFunctorFactory::Instance().Create(value_type, pp);
			//	if( value_type == "Constant" ) {
			//		pfl[pname] = std::shared_ptr<TianXin::GeneralFunctor<EvaluationType>>(new TianXin::ConstantFunctor<EvaluationType>(pll));
			//	}
			//	else if( value_type == "Table" ) {
			//	const auto& p2 = pp.sublist("Table");
			//	_dataT.emplace( pname, std::make_shared<ConstantParamter<T>>(new ConstantParamter<T>(p2)) );
			//	}
			}
			catch (std::exception& e) {
				std::cout << e.what() << std::endl;
			}
		}
		pfls[nname] = pfl;
	}
  }

  template<typename EvaluationType>
  Teuchos::RCP<panzer::ScalarParameterEntry<EvaluationType> >
  accessScalarParameter(const std::string name, panzer::ParamLib& pl)
  {
    Teuchos::RCP<Sacado::ScalarParameterEntry<EvaluationType,panzer::SacadoScalarTypeTraits> > sacado_entry =
      pl.getEntry<EvaluationType>(name);
    return Teuchos::rcp_dynamic_cast<panzer::ScalarParameterEntry<EvaluationType> >(sacado_entry,true);
  }

}

#endif
