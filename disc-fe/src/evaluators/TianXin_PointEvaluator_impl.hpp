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

#ifndef _POINT_EVALUATOR_IMPL_HPP
#define _POINT_EVALUATOR_IMPL_HPP

#include "Panzer_GlobalEvaluationDataContainer.hpp"

#include <set>
#include <stdexcept>

namespace TianXin {
	
template<typename EvalT,typename Traits>
PointEvaluatorBase<EvalT, Traits>::PointEvaluatorBase(const Teuchos::ParameterList& p, const Teuchos::RCP<const TianXin::AbstractDiscretation>& mesh,
      const Teuchos::RCP<const panzer::GlobalIndexer>& indexer)
: PHX::EvaluatorWithBaseImpl<Traits>("Dirichlet Boundary Conditions")
{
    Teuchos::ParameterList params(p);

    std::string eval_name = p.name();  // Dirichlet or CLoad
	m_dof_name = params.get< Teuchos::Array<std::string> >("DOF Names",Teuchos::tuple<std::string>(""));
	try {
		if( m_dof_name.empty() )
			throw std::runtime_error("error in PointEvaluator defintion. DOF Names not given!");
	}
	catch (std::exception& e) {
		 std::cout << e.what() << std::endl;
	}
	m_value_type = params.get<std::string>("Value Type","Constant");
	this->m_pFunctor = WorksetFunctorFactory::Instance().Create(m_value_type, params);

    const auto& s_set_name = params.get<std::string>("SideSet Name","");
	const auto& n_set_name = params.get<std::string>("NodeSet Name","");
	const auto& e_set_name = params.get<std::string>("EdgeSet Name","");
	const auto& f_set_name = params.get<std::string>("FaceSet Name","");
	const auto& eblock_name = params.get<std::string>("ElementSet Name","");

    m_sideset_rank = 3;	
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
			throw std::runtime_error("error in PointEvaluator defintion. Sideset name not given!");
		}
	}
	catch (std::exception& e) {
		 std::cout << e.what() << std::endl;
	}
	for(const auto& myname: m_dof_name )
		eval_name = eval_name+myname;
	eval_name = eval_name+m_sideset_name;
    
    m_penalty = params.get<double>("Penalty",1.e30);
    m_group_id = params.get<int>("Group ID",0);

    std::set< panzer::LocalOrdinal > localIDs;
	std::vector<std::size_t> entities;
    if( m_sideset_rank==0 ) {
		if( eblock_name.empty() )
			mesh->getMyNodeSetIds(m_sideset_name,entities);
		else
			mesh->getMyNodeSetIds(m_sideset_name,eblock_name,entities);
		
		for(auto myname: m_dof_name) {
			int fdnum = indexer->getFieldNum(myname);
			for ( auto nd: entities ) {
				auto b = indexer->getNodalLDofOfField( fdnum, nd );
				if( b<0 ) std::cout << fdnum << ", " << nd <<std::endl;
					TEUCHOS_TEST_FOR_EXCEPTION( (b<0), std::logic_error,
				    "Error - Cannot find dof of Nodeset!" );
				localIDs.insert(b);
			}
		}
	} else if( m_sideset_rank==1 ) {
		if( eblock_name.empty() )
			mesh->getAllEdgeSetIds(m_sideset_name,entities);
		else
			mesh->getAllEdgeSetIds(m_sideset_name,eblock_name,entities);
		for(auto myname: m_dof_name) {
			int fdnum = indexer->getFieldNum(myname);
			for ( auto nd: entities ) {
				auto b = indexer->getEdgeLDofOfField( fdnum, nd );
				if( b.empty() ) std::cout << fdnum << ", " << nd <<std::endl;
				TEUCHOS_TEST_FOR_EXCEPTION( (b.empty()), std::logic_error,
				    "Error - Cannot find dof of Edgeset!" );
				for( const auto ab: b )
					localIDs.insert(ab);
			}
		}
	} else if( m_sideset_rank==-1 ) {
		if( eblock_name.empty() )
			mesh->getMySideSetIds(m_sideset_name,entities);
		else
			mesh->getMySideSetIds(m_sideset_name,eblock_name,entities);
		auto dim = mesh->getDimension();
		for(auto myname: m_dof_name) {
			int fdnum = indexer->getFieldNum(myname);
			for ( auto nd: entities ) {
				//std::cout << indexer->getComm()->getRank()<< "," << fdnum << ", " << nd << std::endl;
				if( dim==2 ) {
					std::vector<panzer::LocalOrdinal> vecb = indexer->getEdgeLDofOfField( fdnum, nd );
					if( vecb.empty() ) std::cout << fdnum << ", " << nd <<std::endl;
						TEUCHOS_TEST_FOR_EXCEPTION( (vecb.empty()), std::logic_error,
						"Error - Cannot find dof of Edgeset!" );
					for( const auto ab : vecb )
						localIDs.insert(ab);	
				}
				else if( dim==1 ) {
					const auto b = indexer->getNodalLDofOfField( fdnum, nd );
					if( b<0 ) std::cout << fdnum << ", " << nd <<std::endl;
						TEUCHOS_TEST_FOR_EXCEPTION( (b<0), std::logic_error,
						"Error - Cannot find dof of Edgeset!" );
					localIDs.insert(b);
				}
			}
		}
	}
	m_ndofs = localIDs.size();
	
	// Create a view in the default execution space
//	Kokkos::View<panzer::LocalOrdinal*,Kokkos::HostSpace> localIDs_k("Dirichelt::localIDs_",m_ndofs);
	// Create a mirror of localIDs_k in host memory
	std::string dof_label = p.name() + "::localIDs_";
	m_local_dofs = Kokkos::View<panzer::LocalOrdinal*,Kokkos::HostSpace>(dof_label,m_ndofs);
	auto localIDs_h = Kokkos::create_mirror_view(m_local_dofs);
	std::size_t nid=0;
	for( const auto& lid: localIDs ) {
         localIDs_h(nid) = lid;
		 ++nid;
    }
	// Copy data from host to device if necessary
    Kokkos::deep_copy(m_local_dofs, localIDs_h);
    // store in Kokkos type
    //m_local_dofs = localIDs_k;

    std::string value_label = p.name() + "::Value_";
	m_values = Kokkos::View<RealType*,Kokkos::HostSpace>(value_label,m_ndofs);

    Teuchos::RCP<PHX::DataLayout> dummy = Teuchos::rcp(new PHX::MDALayout<void>(0));
    const PHX::Tag<ScalarT> fieldTag(eval_name, dummy);

    this->addEvaluatedField(fieldTag);
    this->setName(eval_name+PHX::print<EvalT>());
}

template<typename EvalT,typename Traits>
void PointEvaluatorBase<EvalT, Traits> :: setValues(const panzer::Workset& wk)
{
    // Create a mirror of localIDs_k in host memory
	auto value_h = Kokkos::create_mirror_view(m_values);
	std::size_t nid=0;
	for( std::size_t i=0; i<m_ndofs; ++i ) {
	     value_h(nid) = (*m_pFunctor)(wk);
		 ++nid;
	}
	// Copy data from host to device if necessary
	Kokkos::deep_copy(m_values, value_h);
}

template<typename EvalT,typename Traits>
void PointEvaluatorBase<EvalT, Traits> :: preEvaluate(typename Traits::PreEvalData d)
{
    if(Teuchos::is_null(m_GhostedContainer))
        m_GhostedContainer = Teuchos::rcp_dynamic_cast<panzer::LinearObjContainer>(d.gedc->getDataObject("Ghosted Container"));
}

}

#endif
