// @HEADER
// ***********************************************************************
//
//           TianXin: A partial differential equation assembly
//       engine for strongly coupled complex multiphysics systems
//                 Copyright (2022) Xi Yuan
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
// THIS SOFTWARE IS PROVIDED THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// ***********************************************************************
// @HEADER

#ifndef _TIANXIN_MATERIALBASE_IMPL_HPP
#define _TIANXIN_MATERIALBASE_IMPL_HPP


#include <iostream>
#include <exception>
#include <stdexcept>

namespace TianXin {

	
template< typename T >
MaterialBase<T>::MaterialBase(const Teuchos::ParameterList& params)
{
	_name = params.name();
	for(auto it = params.begin(); it != params.end(); ++it) {
		try {
			const auto& ppl = params.sublist(it->first);
			Teuchos::ParameterList pl(ppl);
			std::string pname = pl.name();
			auto& value_type = pl.get<std::string>("Value Type","Constant");
			if( value_type == "Constant" ) {
				//const auto& p2 = pl.sublist("Constant");
				std::shared_ptr<GeneralParameter<T>> pC(new ConstantParamter<T>(pl));
				_dataT.insert( std::make_pair(pname, pC) );
			}
			else if( value_type == "Table" ) {
			//	const auto& p2 = pl.sublist("Table");
			//	_dataT.emplace( pname, std::make_shared<ConstantParamter<T>>(new ConstantParamter<T>(p2)) );
			}
		}
		catch (std::exception& e) {
			std::cout << e.what() << std::endl;
		}
    /**/
	}
}

template< typename T >
std::vector<T> MaterialBase<T>::eval(std::string name, std::initializer_list<T> independent) const
{
    auto& pfunc = this->_dataT.at(name);
	return (*pfunc)(independent);
}

template< typename T >
void MaterialBase<T>::print(std::ostream& os) const
{
    for( const auto& itr : _dataT ) {
        os << "    Property Name = " << itr.first << std::endl;
    }
}


/*template< typename T, typename... Args >
MaterialBase<T, Args...>::MaterialBase(const Teuchos::ParameterList& params)
{
    for(auto it = params.begin(); it != params.end(); ++it) {
		try {
			const auto& pl = params.sublist(it->first);
		//	items[it->first] = std::make_shared<XYZLib::variable<double>>(pl);
		}
		catch (std::exception& e) {
			std::cout << e.what() << std::endl;
		}
    }
}*/


}

#endif