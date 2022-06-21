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

#include <Teuchos_ConfigDefs.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include "TianXin_MaterialBase.hpp"

namespace TianXin {

	TEUCHOS_UNIT_TEST(material, default)
	{
		Teuchos::ParameterList plmain("Fe");
		{	
			Teuchos::ParameterList& p0 = plmain.sublist("Density");  
			p0.set("Value Type","Constant");
			Teuchos::ParameterList pl_sub("Constant");
			pl_sub.set<Teuchos::Array<double> >("Value",Teuchos::tuple<double>( 10.0 ));
			p0.set("Constant",pl_sub);
	   
			Teuchos::ParameterList& p1 = plmain.sublist("Elasticity");  
			p1.set("Value Type","Constant");
			Teuchos::ParameterList pl_sub1("Constant");
			pl_sub1.set<Teuchos::Array<double> >("Value",Teuchos::tuple<double>( 1000.0,0.3 ));
			p1.set("Constant",pl_sub1);
		}
		//std::cout << "**Name=" << plmain.name() << std::endl;
		//plmain.print();
		
		TianXin::MaterialBase<double> matl(plmain);
		//matl.print();
		
		TEST_EQUALITY(matl.find("Density"), true);
		TEST_EQUALITY(matl.find("Elasticity"), true);
		
		const auto& a = matl.eval("Density");
		TEST_EQUALITY(a[0], 10.0);
		const auto& b = matl.eval("Elasticity");
		TEST_EQUALITY(b[0], 1000.0);
		TEST_EQUALITY(b[1], 0.3);
	}

}