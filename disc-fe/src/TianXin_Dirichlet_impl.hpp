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

#ifndef _BC_DIRICHLET_IMPL_HPP
#define _BC_DIRICHLET_IMPL_HPP

#include <stdexcept>

namespace TianXin {

//**********************************************************************
template<typename EvalT>
Dirichlet<EvalT>::Dirichlet(const Teuchos::ParameterList& params )
: m_group_id(0), m_strategy(0), m_sideset_rank(0)
{
  // ********************
  // Validate and parse parameter list
  // ********************
  {    
    Teuchos::ParameterList valid_params;
    valid_params.set<std::string>("Strategy", "1_0");
    valid_params.set<std::string>("NodeSet Name", "");
    valid_params.set<std::string>("EdgeSet Name", "");
	valid_params.set<std::string>("FaceSet Name", "");
	valid_params.set<std::string>("SideSet Name", "");
    valid_params.set<int>("Group ID", 0);
    valid_params.set<double>("Penalty", 1.e30);
    valid_params.set< Teuchos::Array<std::string> >("DOF Names", Teuchos::tuple<std::string>("") );
    valid_params.set<std::string>("Value Name", "");
  }
  
	m_dof_name = params.get<std::string>("DOF Names");
	try {
		if( m_dof_name.empty() )
			throw std::runtime_error("error in Dirichlet condition defintion. DOF Names not given!");
	}
	catch (std::exception& e) {
		 std::cout << e.what() << std::endl;
	}
	m_value = params.get<std::string>("Value Name");
	try {
		if( m_value.empty() )
			throw std::runtime_error("error in Dirichlet condition defintion. Value Name not given!");
	}
	catch (std::exception& e) {
		 std::cout << e.what() << std::endl;
	}

    const auto& method = params.get<std::string>("Strategy");
    if( method=="Penalty" )
		m_strategy = DiricheltStrategy : Penalty;
	else if( method=="Lagrarangian" )
		m_strategy = DiricheltStrategy : Lagrarangian;
	else
		m_strategy = DiricheltStrategy : 1_0;
    m_strategy = DiricheltStrategy : 1_0;    // only this one is available currently

    const auto& s_set_name = params.get<std::string>("SideSet Name");
	const auto& n_set_name = params.get<std::string>("NodeSet Name");
	const auto& e_set_name = params.get<std::string>("EdgeSet Name");
	const auto& f_set_name = params.get<std::string>("FaceSet Name");
	
	try {
		if( !s_set_name.empty() ) {
			m_sideset_name=s_set_name; m_sideset_rank =-1;
		}
		else if( !n_set_name.empty() ) {
			m_sideset_name=n_set_name; m_sideset_rank =0;
		}
		else if( !e_set_name.empty() ) {
			m_sideset_name=e_set_name; m_sideset_rank =1;
		}
		else if( !f_set_name.empty() ) {
			m_sideset_name=f_set_name; m_sideset_rank =2;
		}
		else {
			throw std::runtime_error("error in Dirichlet condition defintion. Sideset name not given!");
		}
	}
	catch (std::exception& e) {
		 std::cout << e.what() << std::endl;
	}
    
    m_penalty = params.get<double>("Penalty");
    m_group_id = params.get<int>("Group ID");
    
};

}

#endif
