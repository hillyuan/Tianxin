#ifndef __MiniEM_ClosureModelFactory_hpp__
#define __MiniEM_ClosureModelFactory_hpp__

#include "Panzer_ClosureModel_Factory.hpp"

namespace panzer {
  class InputEquationSet;
}

namespace mini_em {

template<typename EvalT>
class ModelFactory : public panzer::ClosureModelFactory<EvalT> {

public:

   std::vector< Teuchos::RCP<PHX::Evaluator<panzer::Traits> > >
   buildClosureModels(const std::string& model_id,
                      const Teuchos::ParameterList& models,
		      const panzer::FieldLayoutLibrary& fl,
		      const Teuchos::RCP<panzer::IntegrationRule>& ir,
                      const Teuchos::ParameterList& default_params,
                      const Teuchos::ParameterList& user_data,
		      const Teuchos::RCP<panzer::GlobalData>& global_data,
                      PHX::FieldManager<panzer::Traits>& fm) const;

   std::vector< Teuchos::RCP<PHX::Evaluator<panzer::Traits> > >
   buildMaterialModels(const Teuchos::RCP<panzer::IntegrationRule>& ir,
                  const std::string& material_name,
                  const std::vector<std::string>& entry_names,
                  const Teuchos::RCP<panzer::GlobalData>& global_data,
                  PHX::FieldManager<panzer::Traits>& fm) const
   {
        std::vector< Teuchos::RCP<PHX::Evaluator<panzer::Traits> > >  evaluators;
        return evaluators;
   }

};

}

#include "MiniEM_ClosureModel_Factory_impl.hpp"

#endif
