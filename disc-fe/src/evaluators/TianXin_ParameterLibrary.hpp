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

#ifndef _TIANXIN_PARAMETERLIBRARY_HPP
#define _TIANXIN_PARAMETERLIBRARY_HPP

#include "TianXin_Parameter.hpp"
#include <unordered_map>
#include <string>
#include <initializer_list>
#include <vector>
#include <functional>
#include <memory>

namespace TianXin {

template< typename T >
struct ParameterLibrary
{
	ParameterLibrary(const Teuchos::ParameterList& params);
	
	/* Material Name */
	std::string _name;
	/* Parameter name and its value */
	std::unordered_map<std::string, std::shared_ptr< TianXin::GeneralParameter<T> > > _dataT;
	
	bool find(const std::string name) const {return _dataT.find(name)!=_dataT.end();}
	std::vector<T> eval(const std::string name, std::initializer_list<T> independent) const;
	std::vector<T> eval(const std::string name) const 
	{
		return this->eval( name,std::initializer_list<T>({}) );
	}
	void print(std::ostream& os = std::cout) const;
};

}

#include "TianXin_ParameterLibrary_impl.hpp"

#endif
