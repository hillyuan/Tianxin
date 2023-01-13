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

#ifndef   __Panzer_LinearObjFactory_hpp__
#define   __Panzer_LinearObjFactory_hpp__

// Panzer
#include "Panzer_CloneableEvaluator.hpp"
#include "Panzer_LinearObjContainer.hpp"
#include "Panzer_ReadOnlyVector_GlobalEvaluationData.hpp"
#include "Panzer_WriteVector_GlobalEvaluationData.hpp"

// Phalanx
#include "Phalanx_Evaluator.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_TemplateManager.hpp"

// Teuchos
#include "Teuchos_DefaultMpiComm.hpp"

// #include "Sacado_mpl_placeholders.hpp"
// using namespace Sacado::mpl::placeholders;

namespace panzer {

class GlobalIndexer; // forward declaration

/** Abstract factory that builds the linear algebra 
  * objects required for the assembly including the 
  * gather/scatter evaluator objects. 
  *
  * The interface for construction of the gather scatter
  * is externally very simple, but in fact under the hood
  * it is quite complex. The user of this factory object
  * simply calls 
     <code>buildGather(const Teuchos::ParameterList & pl) const</code>,
     <code>buildScatter(const Teuchos::ParameterList & pl) const</code>, or
  *
  * To implement a version of this class an author must overide all 
  * the linear algebra construction functions. The new version should also
  * call the base class version of <code>buildGatherScatterEvaluators</code>
  * in the constructor with a reference to itself passed in as an argument.
  * This requires the new version of the class to implement the following functions
      \code
         template <typename EvalT> Teuchos::RCP<panzer::CloneableEvaluator> buildScatter() const;
         template <typename EvalT> Teuchos::RCP<panzer::CloneableEvaluator> buildGather() const;
         template <typename EvalT> Teuchos::RCP<panzer::CloneableEvaluator> buildGatherDomain() const;
         template <typename EvalT> Teuchos::RCP<panzer::CloneableEvaluator> buildGatherOrientation() const;
      \endcode
  * This builds the correct scatter/gather/scatter-dirichlet evaluator objects and returns
  * them as a <code>CloneableEvaluator</code> (These evaluators must overide the <code>CloneableEvaluator::clone</code>
  * function which takes a parameter list). The cloned evaluators will be the ones
  * actually returned from the 
     <code>buildGather(const Teuchos::ParameterList & pl) const</code>,
     <code>buildGatherDomain(const Teuchos::ParameterList & pl) const</code>,
     <code>buildGatherOrientation(const Teuchos::ParameterList & pl) const</code>,
     <code>buildScatter(const Teuchos::ParameterList & pl) const</code>, or
  * functions.
  */
template <typename Traits>
class LinearObjFactory {
public:
    virtual ~LinearObjFactory() {}

    /** This builds all the required evaluators. It is required to be called
      * before the <code>build[Gather,Scatter,ScatterDirichlet]</code> functions
      * are called. This would typically be called by the inheriting class.
      *
      * \param[in] builder Template class to build all required
      *                    evaluators. The class has the following
      *                    interface.
      \code
         template <typename EvalT> Teuchos::RCP<panzer::CloneableEvaluator> buildScatter() const;
         template <typename EvalT> Teuchos::RCP<panzer::CloneableEvaluator> buildGather() const;
         template <typename EvalT> Teuchos::RCP<panzer::CloneableEvaluator> buildGatherDomain() const;
         template <typename EvalT> Teuchos::RCP<panzer::CloneableEvaluator> buildGatherOrientation() const;
      \endcode
      */
    template <typename BuilderT>
    void buildGatherScatterEvaluators(const BuilderT & builder);

   /** Read in a vector from a file. Fill a particular vector in the linear object container.
     *
     * \param[in] identifier Key for specifying which file(s) to read
     * \param[in] loc Linear object container to fill with the vector
     * \param[in] id Id for the field to be filled
     */
    virtual void readVector(const std::string & identifier,LinearObjContainer & loc,int id) const = 0;

   /** Write in a vector from a file. Fill a particular vector in the linear object container.
     *
     * \param[in] identifier Key for specifying which file(s) to read
     * \param[in] loc Linear object container to fill with the vector
     * \param[in] id Id for the field to be filled
     */
    virtual void writeVector(const std::string & identifier,const LinearObjContainer & loc,int id) const = 0;

   /** Build a container with all the neccessary linear algebra objects. This is
     * the non-ghosted version.
     */ 
   virtual Teuchos::RCP<LinearObjContainer> buildLinearObjContainer() const = 0;

   /** Build a container with all the neccessary linear algebra objects, purely on
     * the single physics. This gives linear algebra objects that are relevant for a
     * single physics solve. In many cases this is simply a call to buildLinearObjContainer
     * however, in a few important cases (for instance in stochastic galerkin methods)
     * this will return a container for a single instantiation of the physics. This is
     * the non-ghosted version.
     */ 
   virtual Teuchos::RCP<LinearObjContainer> buildPrimitiveLinearObjContainer() const = 0;

   /** Build a container with all the neccessary linear algebra objects. This is
     * the ghosted version.
     */ 
   virtual Teuchos::RCP<LinearObjContainer> buildGhostedLinearObjContainer() const = 0;

   /** Build a container with all the neccessary linear algebra objects, purely on
     * the single physics. This gives linear algebra objects that are relevant for a
     * single physics solve. In many cases this is simply a call to buildGhostedLinearObjContainer
     * however, in a few important cases (for instance in stochastic galerkin methods)
     * this will return a container for a single instantiation of the physics. This is
     * the ghosted version.
     */ 
   virtual Teuchos::RCP<LinearObjContainer> buildPrimitiveGhostedLinearObjContainer() const = 0;

   /** Build a GlobalEvaluationDataContainer that handles all domain communication.
     * This is used primarily for gather operations and hides the allocation and usage
     * of the ghosted vector from the user.
     */
   virtual Teuchos::RCP<ReadOnlyVector_GlobalEvaluationData> buildReadOnlyDomainContainer() const = 0;

   virtual void globalToGhostContainer(const LinearObjContainer & container,
                                       LinearObjContainer & ghostContainer,int) const = 0;
   virtual void ghostToGlobalContainer(const LinearObjContainer & ghostContainer,
                                       LinearObjContainer & container,int) const = 0;

   /** Initialize container with a specific set of member values.
     *
     * \note This will overwrite everything in the container and zero out values
     *       not requested.
     */
   virtual void initializeContainer(int,LinearObjContainer & loc) const = 0;

   /** Initialize container with a specific set of member values.
     *
     * \note This will overwrite everything in the container and zero out values
     *       not requested.
     */
   virtual void initializeGhostedContainer(int,LinearObjContainer & loc) const = 0;

   /** Acess to the MPI Comm used in constructing this LOF.
     */
   virtual Teuchos::MpiComm<int> getComm() const = 0;

   //! Use preconstructed scatter evaluators
   template <typename EvalT>
   Teuchos::RCP<PHX::Evaluator<Traits> > buildScatter(const Teuchos::ParameterList & pl) const
   { return Teuchos::rcp_dynamic_cast<PHX::Evaluator<Traits> >(scatterManager_->template getAsBase<EvalT>()->clone(pl)); }

   //! Use preconstructed gather evaluators
   template <typename EvalT>
   Teuchos::RCP<PHX::Evaluator<Traits> > buildGather(const Teuchos::ParameterList & pl) const
   { return Teuchos::rcp_dynamic_cast<PHX::Evaluator<Traits> >(gatherManager_->template getAsBase<EvalT>()->clone(pl)); }

   //! Use preconstructed gather evaluators
   template <typename EvalT>
   Teuchos::RCP<PHX::Evaluator<Traits> > buildGatherTangent(const Teuchos::ParameterList & pl) const
   { return Teuchos::rcp_dynamic_cast<PHX::Evaluator<Traits> >(gatherTangentManager_->template getAsBase<EvalT>()->clone(pl)); }

   //! Use preconstructed gather evaluators
   template <typename EvalT>
   Teuchos::RCP<PHX::Evaluator<Traits> > buildGatherDomain(const Teuchos::ParameterList & pl) const
   { return Teuchos::rcp_dynamic_cast<PHX::Evaluator<Traits> >(gatherDomainManager_->template getAsBase<EvalT>()->clone(pl)); }

   //! Use preconstructed gather evaluators
   template <typename EvalT>
   Teuchos::RCP<PHX::Evaluator<Traits> > buildGatherOrientation(const Teuchos::ParameterList & pl) const
   { return Teuchos::rcp_dynamic_cast<PHX::Evaluator<Traits> >(gatherOrientManager_->template getAsBase<EvalT>()->clone(pl)); }


   //! Get the domain global indexer object associated with this factory
   virtual Teuchos::RCP<const panzer::GlobalIndexer> getDomainGlobalIndexer() const = 0;

   //! Get the range global indexer object associated with this factory
   virtual Teuchos::RCP<const panzer::GlobalIndexer> getRangeGlobalIndexer() const = 0;

   virtual void beginFill(LinearObjContainer & /* loc */) const {}
   virtual void endFill(LinearObjContainer & /* loc */) const {}

private:
   typedef PHX::TemplateManager<typename Traits::EvalTypes,
                                panzer::CloneableEvaluator,
                                PHX::EvaluatorDerived<_,Traits> > 
           Evaluator_TemplateManager;

   // managers to build the scatter/gather evaluators
   Teuchos::RCP<Evaluator_TemplateManager> scatterManager_;
   Teuchos::RCP<Evaluator_TemplateManager> gatherManager_;
   Teuchos::RCP<Evaluator_TemplateManager> gatherTangentManager_;
   Teuchos::RCP<Evaluator_TemplateManager> gatherDomainManager_;
   Teuchos::RCP<Evaluator_TemplateManager> gatherOrientManager_;

   template <typename BuilderT>
   struct Scatter_Builder {
      Teuchos::RCP<const BuilderT> builder_;

      Scatter_Builder(const Teuchos::RCP<const BuilderT> & builder) 
         : builder_(builder) {}
      
      template <typename EvalT> Teuchos::RCP<panzer::CloneableEvaluator> build() const 
      { return builder_->template buildScatter<EvalT>(); }
   };

   template <typename BuilderT>
   struct Gather_Builder {
      Teuchos::RCP<const BuilderT> builder_;

      Gather_Builder(const Teuchos::RCP<const BuilderT> & builder) 
         : builder_(builder) {}
     
      template <typename EvalT> Teuchos::RCP<panzer::CloneableEvaluator> build() const 
      { return builder_->template buildGather<EvalT>(); }
   };

   template <typename BuilderT>
   struct GatherTangent_Builder {
      Teuchos::RCP<const BuilderT> builder_;

      GatherTangent_Builder(const Teuchos::RCP<const BuilderT> & builder)
         : builder_(builder) {}

      template <typename EvalT> Teuchos::RCP<panzer::CloneableEvaluator> build() const
      { return builder_->template buildGatherTangent<EvalT>(); }
   };

   template <typename BuilderT>
   struct GatherDomain_Builder {
      Teuchos::RCP<const BuilderT> builder_;

      GatherDomain_Builder(const Teuchos::RCP<const BuilderT> & builder) 
         : builder_(builder) {}
     
      template <typename EvalT> Teuchos::RCP<panzer::CloneableEvaluator> build() const 
      { return builder_->template buildGatherDomain<EvalT>(); }
   };

   template <typename BuilderT>
   struct GatherOrientation_Builder {
      Teuchos::RCP<const BuilderT> builder_;

      GatherOrientation_Builder(const Teuchos::RCP<const BuilderT> & builder) 
         : builder_(builder) {}
     
      template <typename EvalT> Teuchos::RCP<panzer::CloneableEvaluator> build() const 
      { return builder_->template buildGatherOrientation<EvalT>(); }
   };
};

template<typename Traits>
template <typename BuilderT>
inline void LinearObjFactory<Traits>::
buildGatherScatterEvaluators(const BuilderT & builder) 
{
   using Teuchos::rcp;
   using Teuchos::rcpFromRef;

   scatterManager_ = rcp(new Evaluator_TemplateManager);
   scatterManager_->buildObjects(Scatter_Builder<BuilderT>(rcpFromRef(builder)));

   gatherManager_ = Teuchos::rcp(new Evaluator_TemplateManager);
   gatherManager_->buildObjects(Gather_Builder<BuilderT>(rcpFromRef(builder)));

   gatherTangentManager_ = Teuchos::rcp(new Evaluator_TemplateManager);
   gatherTangentManager_->buildObjects(GatherTangent_Builder<BuilderT>(rcpFromRef(builder)));

   gatherDomainManager_ = Teuchos::rcp(new Evaluator_TemplateManager);
   gatherDomainManager_->buildObjects(GatherDomain_Builder<BuilderT>(rcpFromRef(builder)));

   gatherOrientManager_ = Teuchos::rcp(new Evaluator_TemplateManager);
   gatherOrientManager_->buildObjects(GatherOrientation_Builder<BuilderT>(rcpFromRef(builder)));
}

}

#endif // __Panzer_LinearObjFactory_hpp__
