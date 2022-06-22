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

#ifndef _TIANXIN_PRAMETERLIBRARYFACTORY_HPP
#define _TIANXIN_PRAMETERLIBRARYFACTORY_HPP

#include "TianXin_MaterialBase.hpp"
#include "TianXin_ParameterEvaluator.hpp"
#include "Panzer_EvaluatorsRegistrar.hpp"

namespace TianXin {

class ParameterLibraryFactoryBase : public panzer::EvaluatorsRegistrar{

  public:
    Teuchos::RCP< std::vector< Teuchos::RCP<PHX::Evaluator<panzer::Traits> > > >
    virtual buildEvalator(const std::vector< Teuchos::RCP<panzer::PhysicsBlock> > & physicsBlocks,
           std::shared_ptr<TianXin::MaterialBase<double>> materials ) const = 0;

    /** This a convenience function for registering the evaluators. Essentially this
      * facilitates better usage of the ClosureModel TM and allows an easy registration
      * process externally without knowning the compile-time evaluation type.
      *
      * \param[in] evaluators Evaluators to register
      * \param[in] fm Field manager where the evaluators will be registered on completion.
      */
    virtual void registerEvaluators(const std::vector< Teuchos::RCP<PHX::Evaluator<panzer::Traits> > > & evaluators,
                                    PHX::FieldManager<panzer::Traits>& fm) const = 0;

};


template<typename EvalT>
class ParameterLibraryFactory : public ParameterLibraryFactoryBase {

  protected:
    bool m_throw_if_model_not_found;
  public:

    ParameterLibraryFactory(bool throw_if_model_not_found=true) : m_throw_if_model_not_found(throw_if_model_not_found) {}
    
    Teuchos::RCP< std::vector< Teuchos::RCP<PHX::Evaluator<panzer::Traits> > > >
    buildEvalator(const std::vector< Teuchos::RCP<panzer::PhysicsBlock> > & physicsBlocks,
           std::shared_ptr<TianXin::MaterialBase<double>> materials ) const;

    /** This a convenience function for registering the evaluators. Essentially this
      * facilitates better usage of the ClosureModel TM and allows an easy registration
      * process externally without knowning the compile-time evaluation type.
      *
      * \param[in] evaluators Evaluators to register
      * \param[in] fm Field manager where the evaluators will be registered on completion.
      */
    virtual void registerEvaluators(const std::vector< Teuchos::RCP<PHX::Evaluator<panzer::Traits> > > & evaluators,
                                    PHX::FieldManager<panzer::Traits>& fm) const
    { 
      for (std::vector< Teuchos::RCP<PHX::Evaluator<panzer::Traits> > >::size_type i=0; i < evaluators.size(); ++i)
        this->template registerEvaluator<EvalT>(fm, evaluators[i]);
    }

    virtual void setThrowOnModelNotFound(bool do_throw) {
      m_throw_if_model_not_found=do_throw;
    }

};

}

//#include "TianXin_ParameterLibraryFactory_impl.hpp"

#endif
