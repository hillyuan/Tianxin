// @HEADER
// ***********************************************************************
//
//           TianXin: A partial differential equation assembly
//       engine for strongly coupled complex multiphysics systems
//                 Copyright (2022) YUAN Xi
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

#ifndef _TIANXIN_NEUMANN_EVALUATOR_HPP
#define _TIANXIN_NEUMANN_EVALUATOR_HPP

#include "Phalanx_Evaluator_Macros.hpp"
#include "Phalanx_MDField.hpp"

#include "TianXin_WorksetFunctor.hpp"
#include "TianXin_Factory.hpp"

#include <memory>

namespace TianXin {
    
  /** \brief Evaluates a Neumann BC residual contribution

      computes the surface integral term resulting from integration
      by parts for a particular dof:

      int(n \cdot (flux * phi) )
  */
template<typename EvalT, typename Traits>
class NeumannBase : public PHX::EvaluatorWithBaseImpl<Traits>
{
	using ScalarT = typename EvalT::ScalarT;

public:

    NeumannBase(const Teuchos::ParameterList& p);

    void postRegistrationSetup(typename Traits::SetupData d,PHX::FieldManager<Traits>& fm);
    virtual void evaluateFields(typename Traits::EvalData d) = 0;

private:
  
    PHX::MDField<ScalarT> residual;
  
    // output
    //Kokkos::DynRankView<ScalarT, PHX::Device> neumann;

    std::string basis_name;
    std::size_t basis_index;
}; // end of class NeumannBase

typedef Factory<NeumannBase,std::string,Teuchos::ParameterList> NeumannFunctorFactory;

#define REGISTER_NEUMANN(CLASSNAME) \
	static const auto CLASSNAME##register_result = NeumannFunctorFactory::Instance().Register<CLASSNAME>(#CLASSNAME); \

// **************************************************************
// Scalar Flux, e.g. Heat Flux dT/dn
// **************************************************************

template<typename EvalT, typename Traits>
class Flux : public NeumannBase<EvalT, Traits>
{
  public:
    Flux(const Teuchos::ParameterList& params );
    void evaluateFields(typename Traits::EvalData d) final;
  private:
    std::unique_ptr<TianXin::WorksetFunctor> pFunc;
};
namespace NeumannRegister {
	static bool const FLUX_ROK = NeumannFunctorFactory::Instance().template Register< Flux<panzer::Traits::Residual,panzer::Traits> >( "Flux");
	static bool const FLUX_JOK = NeumannFunctorFactory::Instance().template Register< Flux<panzer::Traits::Jacobian,panzer::Traits> >( "Flux");
}


}

#endif
