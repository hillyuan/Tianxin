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

#include "PanzerAdaptersSTK_config.hpp"

#include "TianXin_STK_Utilities.hpp"
#include "Panzer_GlobalIndexer.hpp"

#include "Kokkos_DynRankView.hpp"

#include <stk_mesh/base/FieldBase.hpp>

namespace TianXin {

#ifdef PANZER_HAVE_EPETRA
static void gather_in_block(const std::string & blockId, const panzer::GlobalIndexer& dofMngr,
                            const Epetra_Vector & x,const std::vector<std::size_t> & localCellIds,
                            std::map<std::string,Kokkos::DynRankView<double,PHX::Device> > & fc);
#endif
static void gather_in_block_tpetra(const std::string & blockId, const panzer::GlobalIndexer& dofMngr,
                            const Tpetra::Vector<double,int,panzer::GlobalOrdinal>& x,const std::vector<std::size_t> & localCellIds,
                            std::map<std::string,Kokkos::DynRankView<double,PHX::Device> > & fc);

static void build_local_ids(const panzer_stk::STK_Interface & mesh,
                            std::map<std::string,Teuchos::RCP<std::vector<std::size_t> > > & localIds);

#ifdef PANZER_HAVE_EPETRA
void write_solution_data(const panzer::GlobalIndexer& dofMngr,panzer_stk::STK_Interface & mesh,const Epetra_MultiVector & x,const std::string & prefix,const std::string & postfix)
{
   write_solution_data(dofMngr,mesh,*x(0),prefix,postfix);
}

void write_solution_data(const panzer::GlobalIndexer& dofMngr,panzer_stk::STK_Interface & mesh,const Epetra_Vector & x,const std::string & prefix,const std::string & postfix)
{
   typedef Kokkos::DynRankView<double,PHX::Device> FieldContainer;

   // get local IDs
   std::map<std::string,Teuchos::RCP<std::vector<std::size_t> > > localIds;
   build_local_ids(mesh,localIds);

   // loop over all element blocks
   for(const auto & itr : localIds) {
      const auto blockId = itr.first;
      const auto & localCellIds = *(itr.second);

      std::map<std::string,FieldContainer> data;

      // get all solution data for this block
      gather_in_block(blockId,dofMngr,x,localCellIds,data);

      // write out to stk mesh
      for(const auto & dataItr : data)
         mesh.setSolutionFieldData(prefix+dataItr.first+postfix,blockId,localCellIds,dataItr.second);
   }
}

void gather_in_block(const std::string & blockId, const panzer::GlobalIndexer& dofMngr,
                     const Epetra_Vector & x,const std::vector<std::size_t> & localCellIds,
                     std::map<std::string,Kokkos::DynRankView<double,PHX::Device> > & fc)
{
   const std::vector<int> & fieldNums = dofMngr.getBlockFieldNumbers(blockId);

   for(std::size_t fieldIndex=0;fieldIndex<fieldNums.size();fieldIndex++) {
      int fieldNum = fieldNums[fieldIndex];
      std::string fieldStr = dofMngr.getFieldString(fieldNum);

      // grab the field
      const std::vector<int> & elmtOffset = dofMngr.getGIDFieldOffsets(blockId,fieldNum);
      fc[fieldStr] = Kokkos::DynRankView<double,PHX::Device>("fc",localCellIds.size(),elmtOffset.size());
      auto field = Kokkos::create_mirror_view(fc[fieldStr]);


      // gather operation for each cell in workset
      for(std::size_t worksetCellIndex=0;worksetCellIndex<localCellIds.size();++worksetCellIndex) {
         std::vector<panzer::GlobalOrdinal> GIDs;
         std::vector<int> LIDs;
         std::size_t cellLocalId = localCellIds[worksetCellIndex];

         dofMngr.getElementGIDs(cellLocalId,GIDs);

         // caculate the local IDs for this element
         LIDs.resize(GIDs.size());
         for(std::size_t i=0;i<GIDs.size();i++)
            LIDs[i] = x.Map().LID(GIDs[i]);

         // loop over basis functions and fill the fields
         for(std::size_t basis=0;basis<elmtOffset.size();basis++) {
            int offset = elmtOffset[basis];
            int lid = LIDs[offset];
            field(worksetCellIndex,basis) = x[lid];
         }
      }
      Kokkos::deep_copy(fc[fieldStr], field);
   }
}
#endif

void write_solution_data(const panzer::GlobalIndexer& dofMngr,panzer_stk::STK_Interface & mesh,const Tpetra::Vector<double,panzer::LocalOrdinal,panzer::GlobalOrdinal>& x,const std::string & prefix,const std::string & postfix)
{
   typedef Kokkos::DynRankView<double,PHX::Device> FieldContainer;

   // get local IDs
   std::map<std::string,Teuchos::RCP<std::vector<std::size_t> > > localIds;
   build_local_ids(mesh,localIds);

   // loop over all element blocks
   for(const auto & itr : localIds) {
      const auto blockId = itr.first;
      const auto & localCellIds = *(itr.second);

      std::map<std::string,FieldContainer> data;

      // get all solution data for this block
      gather_in_block_tpetra(blockId,dofMngr,x,localCellIds,data);

      // write out to stk mesh
      for(const auto & dataItr : data)
         mesh.setSolutionFieldData(prefix+dataItr.first+postfix,blockId,localCellIds,dataItr.second);
   }
}

void gather_in_block_tpetra(const std::string & blockId, const panzer::GlobalIndexer& dofMngr,
                     const Tpetra::Vector<double,int,panzer::GlobalOrdinal>& xvec,
					 const std::vector<std::size_t> & localCellIds,
                     std::map<std::string,Kokkos::DynRankView<double,PHX::Device> > & fc)
{
   const std::vector<int> & fieldNums = dofMngr.getBlockFieldNumbers(blockId);
   auto x= xvec.getData();

   for(std::size_t fieldIndex=0;fieldIndex<fieldNums.size();fieldIndex++) {
      int fieldNum = fieldNums[fieldIndex];
      std::string fieldStr = dofMngr.getFieldString(fieldNum);

      // grab the field
      const std::vector<int> & elmtOffset = dofMngr.getGIDFieldOffsets(blockId,fieldNum);
      fc[fieldStr] = Kokkos::DynRankView<double,PHX::Device>("fc",localCellIds.size(),elmtOffset.size());
      auto field = Kokkos::create_mirror_view(fc[fieldStr]);


      // gather operation for each cell in workset
      for(std::size_t worksetCellIndex=0;worksetCellIndex<localCellIds.size();++worksetCellIndex) {
         std::vector<panzer::GlobalOrdinal> GIDs;
         std::vector<int> LIDs;
         std::size_t cellLocalId = localCellIds[worksetCellIndex];

         dofMngr.getElementGIDs(cellLocalId,GIDs);

         // caculate the local IDs for this element
         LIDs.resize(GIDs.size());
         for(std::size_t i=0;i<GIDs.size();i++)
            LIDs[i] = xvec.getMap()->getLocalElement(GIDs[i]);

         // loop over basis functions and fill the fields
         for(std::size_t basis=0;basis<elmtOffset.size();basis++) {
            int offset = elmtOffset[basis];
            int lid = LIDs[offset];
            field(worksetCellIndex,basis) = x[lid];
         }
      }
      Kokkos::deep_copy(fc[fieldStr], field);
   }
}

void build_local_ids(const panzer_stk::STK_Interface & mesh,
                   std::map<std::string,Teuchos::RCP<std::vector<std::size_t> > > & localIds)
{
   // defines ordering of blocks
   std::vector<std::string> blockIds;
   mesh.getElementBlockNames(blockIds);

   std::vector<std::string>::const_iterator idItr;
   for(idItr=blockIds.begin();idItr!=blockIds.end();++idItr) {
      std::string blockId = *idItr;

      localIds[blockId] = Teuchos::rcp(new std::vector<std::size_t>);
      std::vector<std::size_t> & localBlockIds = *localIds[blockId];

      // grab elements on this block
      std::vector<stk::mesh::Entity> blockElmts;
      mesh.getMyElements(blockId,blockElmts);

      std::vector<stk::mesh::Entity>::const_iterator itr;
      for(itr=blockElmts.begin();itr!=blockElmts.end();++itr)
         localBlockIds.push_back(mesh.elementLocalId(*itr));

      std::sort(localBlockIds.begin(),localBlockIds.end());
   }
}

}
