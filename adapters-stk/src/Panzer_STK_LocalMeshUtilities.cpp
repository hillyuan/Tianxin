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

#include "Panzer_NodeType.hpp"
#include "Panzer_STK_LocalMeshUtilities.hpp"
#include "Panzer_STK_Interface.hpp"
#include "Panzer_STK_SetupUtilities.hpp"
#include "Panzer_STKConnManager.hpp"

#include "Panzer_HashUtils.hpp"
#include "Panzer_LocalMeshInfo.hpp"
#include "Panzer_LocalPartitioningUtilities.hpp"
#include "Panzer_FaceToElement.hpp"

#include "Panzer_FieldPattern.hpp"
#include "Panzer_NodalFieldPattern.hpp"
#include "Panzer_EdgeFieldPattern.hpp"
#include "Panzer_FaceFieldPattern.hpp"
#include "Panzer_ElemFieldPattern.hpp"

#include "Panzer_ConnManager.hpp"

#include "Phalanx_KokkosDeviceTypes.hpp"

#include "Teuchos_Assert.hpp"
#include "Teuchos_OrdinalTraits.hpp"

#include "Tpetra_Import.hpp"

#include <string>
#include <map>
#include <vector>
#include <unordered_set>

namespace panzer_stk
{

// No external access
namespace
{


/** This method takes a cell importer (owned to ghstd) and communicates vertices
  * of the ghstd elements.
  */
Kokkos::DynRankView<double,PHX::Device>
buildGhostedVertices(const Tpetra::Import<int,panzer::GlobalOrdinal,panzer::TpetraNodeType> & importer,
                     Kokkos::DynRankView<const double,PHX::Device> owned_vertices)
{
  using Teuchos::RCP;
  using Teuchos::rcp;

  typedef Tpetra::MultiVector<double,int,panzer::GlobalOrdinal,panzer::TpetraNodeType> mvec_type;
  typedef typename mvec_type::dual_view_type dual_view_type;

  size_t owned_cell_cnt = importer.getSourceMap()->getNodeNumElements();
  size_t ghstd_cell_cnt = importer.getTargetMap()->getNodeNumElements();
  int vertices_per_cell = owned_vertices.extent(1);
  int space_dim         = owned_vertices.extent(2);

  TEUCHOS_ASSERT(owned_vertices.extent(0)==owned_cell_cnt);

  // build vertex multivector
  RCP<mvec_type> owned_vertices_mv   = rcp(new mvec_type(importer.getSourceMap(),vertices_per_cell*space_dim));
  RCP<mvec_type> ghstd_vertices_mv = rcp(new mvec_type(importer.getTargetMap(),vertices_per_cell*space_dim));

  {
    auto owned_vertices_view = owned_vertices_mv->template getLocalView<dual_view_type>();
    for(size_t i=0;i<owned_cell_cnt;i++) {
      int l = 0;
      for(int j=0;j<vertices_per_cell;j++)
        for(int k=0;k<space_dim;k++,l++)
          owned_vertices_view(i,l) = owned_vertices(i,j,k);
    }
  }

  // communicate ghstd vertices
  ghstd_vertices_mv->doImport(*owned_vertices_mv,importer,Tpetra::INSERT);

  // copy multivector into ghstd vertex structure
  Kokkos::DynRankView<double,PHX::Device> ghstd_vertices("ghstd_vertices",ghstd_cell_cnt,vertices_per_cell,space_dim);
  {
    auto ghstd_vertices_view = ghstd_vertices_mv->template getLocalView<dual_view_type>();
    for(size_t i=0;i<ghstd_cell_cnt;i++) {
      int l = 0;
      for(int j=0;j<vertices_per_cell;j++)
        for(int k=0;k<space_dim;k++,l++)
          ghstd_vertices(i,j,k) = ghstd_vertices_view(i,l);
    }
  }

  return ghstd_vertices;
} // end build ghstd vertices


	
void
setupSubLocalMeshInfo(const panzer_stk::STK_Interface & mesh,
					  const panzer::LocalMeshInfoBase & parent_info,
                      const std::vector<panzer::LocalOrdinal> & owned_parent_cells,
                      panzer::LocalMeshInfoBase & sub_info)
{
  using GO = panzer::GlobalOrdinal;
  using LO = panzer::LocalOrdinal;

  PANZER_FUNC_TIME_MONITOR_DIFF("panzer::partitioning_utilities::setupSubLocalMeshInfo",setupSLMI);
  // The goal of this function is to fill a LocalMeshInfoBase (sub_info) with
  // a subset of cells from a given parent LocalMeshInfoBase (parent_info)

  // Note: owned_parent_cells are the owned cells for sub_info in the parent_info's indexing scheme
  // We need to generate sub_info's ghosts and figure out the virtual cells

  // Note: We will only handle a single ghost layer

  // Note: We assume owned_parent_cells are owned cells of the parent
  // i.e. owned_parent_indexes cannot refer to ghost or virtual cells in parent_info

  // Note: This function works with inter-face connectivity. NOT node connectivity.

  const int num_owned_cells = owned_parent_cells.size();
  TEUCHOS_TEST_FOR_EXCEPT_MSG(num_owned_cells == 0, "panzer::partitioning_utilities::setupSubLocalMeshInfo : Input parent subcells must exist (owned_parent_cells)");

  const int num_parent_owned_cells = parent_info.num_owned_cells;
  TEUCHOS_TEST_FOR_EXCEPT_MSG(num_parent_owned_cells <= 0, "panzer::partitioning_utilities::setupSubLocalMeshInfo : Input parent info must contain owned cells");

  const int num_parent_ghstd_cells = parent_info.num_ghstd_cells;
  const int num_parent_total_cells = parent_info.num_owned_cells + parent_info.num_ghstd_cells + parent_info.num_virtual_cells;

  // Just as a precaution, make sure the parent_info is setup properly
  TEUCHOS_ASSERT(static_cast<int>(parent_info.cell_to_faces.extent(0)) == num_parent_total_cells);
  const int num_faces_per_cell = parent_info.cell_to_faces.extent(1);

  // The first thing to do is construct a vector containing the parent cell indexes of all
  // owned, ghstd, and virtual cells
  std::vector<LO> ghstd_parent_cells;
  std::vector<LO> virtual_parent_cells;
  {
    PANZER_FUNC_TIME_MONITOR_DIFF("Construct parent cell vector",ParentCell);
    // We grab all of the owned cells and put their global indexes into sub_info
    // We also put all of the owned cell indexes in the parent's indexing scheme into a set to use for lookups
    std::unordered_set<LO> owned_parent_cells_set(owned_parent_cells.begin(), owned_parent_cells.end());

    // We need to create a list of ghstd and virtual cells
    // We do this by running through sub_cell_indexes
    // and looking at the neighbors to find neighbors that are not owned

    // Virtual cells are defined as cells with indexes outside of the range of owned_cells and ghstd_cells
    const int virtual_parent_cell_offset = num_parent_owned_cells + num_parent_ghstd_cells;

    std::unordered_set<LO> ghstd_parent_cells_set;
    std::unordered_set<LO> virtual_parent_cells_set;
    for(int i=0;i<num_owned_cells;++i){
      const LO parent_cell_index = owned_parent_cells[i];
      for(int local_face_index=0;local_face_index<num_faces_per_cell;++local_face_index){
        const LO parent_face = parent_info.cell_to_faces(parent_cell_index, local_face_index);

        // Sidesets can have owned cells that border the edge of the domain (i.e. parent_face == -1)
        // If we are at the edge of the domain, we can ignore this face.
        if(parent_face < 0)
          continue;

        // Find the side index for neighbor cell with respect to the face
        const LO neighbor_parent_side = (parent_info.face_to_cells(parent_face,0) == parent_cell_index) ? 1 : 0;

        // Get the neighbor cell index in the parent's indexing scheme
        const LO neighbor_parent_cell = parent_info.face_to_cells(parent_face, neighbor_parent_side);

        // If the face exists, then the neighbor should exist
        TEUCHOS_ASSERT(neighbor_parent_cell >= 0);

        // We can easily check if this is a virtual cell
        if(neighbor_parent_cell >= virtual_parent_cell_offset){
          virtual_parent_cells_set.insert(neighbor_parent_cell);
        } else if(neighbor_parent_cell >= num_parent_owned_cells){
          // This is a quick check for a ghost cell
          // This branch only exists to cut down on the number of times the next branch (much slower) is called
          ghstd_parent_cells_set.insert(neighbor_parent_cell);
        } else {
          // There is still potential for this to be a ghost cell with respect to 'our' cells
          // The only way to check this is with a super slow lookup call
          if(owned_parent_cells_set.find(neighbor_parent_cell) == owned_parent_cells_set.end()){
            // The neighbor cell is not owned by 'us', therefore it is a ghost
            ghstd_parent_cells_set.insert(neighbor_parent_cell);
          }
        }
      }
    }

    // We now have a list of the owned, ghstd, and virtual cells in the parent's indexing scheme.
    // We will take the 'unordered_set's ordering for the the sub-indexing scheme

    ghstd_parent_cells.assign(ghstd_parent_cells_set.begin(), ghstd_parent_cells_set.end());
    virtual_parent_cells.assign(virtual_parent_cells_set.begin(), virtual_parent_cells_set.end());

  }

  const int num_ghstd_cells = ghstd_parent_cells.size();
  const int num_virtual_cells = virtual_parent_cells.size();
  const int num_real_cells = num_owned_cells + num_ghstd_cells;
  const int num_total_cells = num_real_cells + num_virtual_cells;

  std::vector<std::pair<LO, LO> > all_parent_cells(num_total_cells);
  for (std::size_t i=0; i< owned_parent_cells.size(); ++i)
    all_parent_cells[i] = std::pair<LO, LO>(owned_parent_cells[i], i);

  for (std::size_t i=0; i< ghstd_parent_cells.size(); ++i) {
    LO insert = owned_parent_cells.size()+i;
    all_parent_cells[insert] = std::pair<LO, LO>(ghstd_parent_cells[i], insert);
  }

  for (std::size_t i=0; i< virtual_parent_cells.size(); ++i) {
    LO insert = owned_parent_cells.size()+ ghstd_parent_cells.size()+i;
    all_parent_cells[insert] = std::pair<LO, LO>(virtual_parent_cells[i], insert);
  }

  sub_info.num_owned_cells = owned_parent_cells.size();
  sub_info.num_ghstd_cells = ghstd_parent_cells.size();
  sub_info.num_virtual_cells = virtual_parent_cells.size();
  // We now have the indexing order for our sub_info

  // Just as a precaution, make sure the parent_info is setup properly
  TEUCHOS_ASSERT(static_cast<int>(parent_info.cell_vertices.extent(0)) == num_parent_total_cells);
  TEUCHOS_ASSERT(static_cast<int>(parent_info.local_cells.extent(0)) == num_parent_total_cells);
  TEUCHOS_ASSERT(static_cast<int>(parent_info.global_cells.extent(0)) == num_parent_total_cells);

  const int num_vertices_per_cell = parent_info.cell_vertices.extent(1);
  const int num_dims = parent_info.cell_vertices.extent(2);

  // Fill owned, ghstd, and virtual cells: global indexes, local indexes and vertices
  sub_info.global_cells = PHX::View<GO*>("global_cells", num_total_cells);
  sub_info.local_cells = PHX::View<LO*>("local_cells", num_total_cells);
  sub_info.cell_vertices = PHX::View<double***>("cell_vertices", num_total_cells, num_vertices_per_cell, num_dims);
  for(int cell=0;cell<num_total_cells;++cell){
    const LO parent_cell = all_parent_cells[cell].first;
    sub_info.global_cells(cell) = parent_info.global_cells(parent_cell);
    sub_info.local_cells(cell) = parent_info.local_cells(parent_cell);
    for(int vertex=0;vertex<num_vertices_per_cell;++vertex){
      for(int dim=0;dim<num_dims;++dim){
        sub_info.cell_vertices(cell,vertex,dim) = parent_info.cell_vertices(parent_cell,vertex,dim);
      }
    }
  }

  // Now for the difficult part

  // We need to create a new face indexing scheme from the old face indexing scheme

  // Create an auxiliary list with all cells - note this preserves indexing

  struct face_t{
    face_t(LO c0, LO c1, LO sc0, LO sc1)
    {
      cell_0=c0;
      cell_1=c1;
      subcell_index_0=sc0;
      subcell_index_1=sc1;
    }
    LO cell_0;
    LO cell_1;
    LO subcell_index_0;
    LO subcell_index_1;
  };


  // First create the faces
  std::vector<face_t> faces;
  {
    PANZER_FUNC_TIME_MONITOR_DIFF("Create faces",CreateFaces);
    // faces_set: cell_0, subcell_index_0, cell_1, subcell_index_1
    std::unordered_map<LO,std::unordered_map<LO, std::pair<LO,LO> > > faces_set;

    std::sort(all_parent_cells.begin(), all_parent_cells.end());

    for(int owned_cell=0;owned_cell<num_owned_cells;++owned_cell){
      const LO owned_parent_cell = owned_parent_cells[owned_cell];
      for(int local_face=0;local_face<num_faces_per_cell;++local_face){
        const LO parent_face = parent_info.cell_to_faces(owned_parent_cell,local_face);

        // Skip faces at the edge of the domain
        if(parent_face<0)
          continue;

        // Get the cell on the other side of the face
        const LO neighbor_side = (parent_info.face_to_cells(parent_face,0) == owned_parent_cell) ? 1 : 0;

        const LO neighbor_parent_cell = parent_info.face_to_cells(parent_face, neighbor_side);
        const LO neighbor_subcell_index = parent_info.face_to_lidx(parent_face, neighbor_side);

        // Convert parent cell index into sub cell index
        std::pair<LO, LO> search_point(neighbor_parent_cell, 0);
        auto itr = std::lower_bound(all_parent_cells.begin(), all_parent_cells.end(), search_point);

        TEUCHOS_TEST_FOR_EXCEPT_MSG(itr == all_parent_cells.end(), "panzer_stk::setupSubLocalMeshInfo : Neighbor cell was not found in owned, ghosted, or virtual cells");

        const LO neighbor_cell = itr->second;

        LO cell_0, cell_1, subcell_index_0, subcell_index_1;
        if(owned_cell < neighbor_cell){
          cell_0 = owned_cell;
          subcell_index_0 = local_face;
          cell_1 = neighbor_cell;
          subcell_index_1 = neighbor_subcell_index;
        } else {
          cell_1 = owned_cell;
          subcell_index_1 = local_face;
          cell_0 = neighbor_cell;
          subcell_index_0 = neighbor_subcell_index;
        }

        // Add this interface to the set of faces - smaller cell index is 'left' (or '0') side of face
        faces_set[cell_0][subcell_index_0] = std::pair<LO,LO>(cell_1, subcell_index_1);
      }
    }

    for(const auto & cell_pair : faces_set){
      const LO cell_0 = cell_pair.first;
      for(const auto & subcell_pair : cell_pair.second){
        const LO subcell_index_0 = subcell_pair.first;
        const LO cell_1 = subcell_pair.second.first;
        const LO subcell_index_1 = subcell_pair.second.second;
        faces.push_back(face_t(cell_0,cell_1,subcell_index_0,subcell_index_1));
      }
    }
  }

  const int num_faces = faces.size();

  sub_info.face_to_cells = PHX::View<LO*[2]>("face_to_cells", num_faces);
  sub_info.face_to_lidx = PHX::View<LO*[2]>("face_to_lidx", num_faces);
  sub_info.cell_to_faces = PHX::View<LO**>("cell_to_faces", num_total_cells, num_faces_per_cell);

  // Default the system with invalid cell index
  Kokkos::deep_copy(sub_info.cell_to_faces, -1);

  for(int face_index=0;face_index<num_faces;++face_index){
    const face_t & face = faces[face_index];

    sub_info.face_to_cells(face_index,0) = face.cell_0;
    sub_info.face_to_cells(face_index,1) = face.cell_1;

    sub_info.cell_to_faces(face.cell_0,face.subcell_index_0) = face_index;
    sub_info.cell_to_faces(face.cell_1,face.subcell_index_1) = face_index;

    sub_info.face_to_lidx(face_index,0) = face.subcell_index_0;
    sub_info.face_to_lidx(face_index,1) = face.subcell_index_1;

  }

}

void
setupLocalMeshBlockInfo(const panzer_stk::STK_Interface & mesh,
                        panzer::ConnManager & conn,
                        const panzer::LocalMeshInfo & mesh_info,
                        const std::string & element_block_name,
                        panzer::LocalMeshBlockInfo & block_info)
{

  // This function identifies all cells in mesh_info that belong to element_block_name
  // and creates a block_info from it.

  const int num_parent_owned_cells = mesh_info.num_owned_cells;

  // Make sure connectivity is setup for interfaces between cells
  {
    const shards::CellTopology & topology = *(mesh.getCellTopology(element_block_name));
    Teuchos::RCP<panzer::FieldPattern> cell_pattern;
    if(topology.getDimension() == 1){
      cell_pattern = Teuchos::rcp(new panzer::EdgeFieldPattern(topology));
    } else if(topology.getDimension() == 2){
      cell_pattern = Teuchos::rcp(new panzer::FaceFieldPattern(topology));
    } else if(topology.getDimension() == 3){
      cell_pattern = Teuchos::rcp(new panzer::ElemFieldPattern(topology));
    }

    {
      PANZER_FUNC_TIME_MONITOR("Build connectivity");
      conn.buildConnectivity(*cell_pattern);
    }
  }

  std::vector<panzer::LocalOrdinal> owned_block_cells;
  for(int parent_owned_cell=0;parent_owned_cell<num_parent_owned_cells;++parent_owned_cell){
    const panzer::LocalOrdinal local_cell = mesh_info.local_cells(parent_owned_cell);
    const bool is_in_block = conn.getBlockId(local_cell) == element_block_name;

    if(is_in_block){
      owned_block_cells.push_back(parent_owned_cell);
    }

  }

  if ( owned_block_cells.size() == 0 )
    return;
  block_info.num_owned_cells = owned_block_cells.size();
  block_info.element_block_name = element_block_name;
  block_info.cell_topology = mesh.getCellTopology(element_block_name);
  {
    PANZER_FUNC_TIME_MONITOR("setupSubLocalMeshInfo");
    setupSubLocalMeshInfo(mesh, mesh_info, owned_block_cells, block_info);
    //PANZER_FUNC_TIME_MONITOR("setupBlockMeshInfo");
    //setupBlockMeshInfo(mesh, mesh_info, owned_block_cells, block_info);
  }
}


void
setupLocalMeshSidesetInfo(const panzer_stk::STK_Interface & mesh,
                          panzer::ConnManager& /* conn */,
                          const panzer::LocalMeshInfo & mesh_info,
                          const std::string & element_block_name,
                          const std::string & sideset_name,
                          panzer::LocalMeshSidesetInfo & sideset_info)
{

  // We use these typedefs to make the algorithm slightly more clear
  using LocalOrdinal = panzer::LocalOrdinal;
  using ParentOrdinal = int;
  using SubcellOrdinal = short;

  // This function identifies all cells in mesh_info that belong to element_block_name
  // and creates a block_info from it.

  // This is a list of all entities that make up the 'side'
  std::vector<stk::mesh::Entity> side_entities;

  // Grab the side entities associated with this sideset on the mesh
  // Note: Throws exception if element block or sideset doesn't exist
  try{

    mesh.getAllSides(sideset_name, element_block_name, side_entities);

  } catch(STK_Interface::SidesetException & e) {
     std::stringstream ss;
     std::vector<std::string> sideset_names;
     mesh.getSidesetNames(sideset_names);

     // build an error message
     ss << e.what() << "\nChoose existing sideset:\n";
     for(const auto & optional_sideset_name : sideset_names){
        ss << "\t\"" << optional_sideset_name << "\"\n";
     }

     TEUCHOS_TEST_FOR_EXCEPTION_PURE_MSG(true,std::logic_error,ss.str());

  } catch(STK_Interface::ElementBlockException & e) {
     std::stringstream ss;
     std::vector<std::string> element_block_names;
     mesh.getElementBlockNames(element_block_names);

     // build an error message
     ss << e.what() << "\nChoose existing element block:\n";
     for(const auto & optional_element_block_name : element_block_names){
        ss << "\t\"" << optional_element_block_name << "\"\n";
     }

     TEUCHOS_TEST_FOR_EXCEPTION_PURE_MSG(true,std::logic_error,ss.str());

  } catch(std::logic_error & e) {
     std::stringstream ss;
     ss << e.what() << "\nUnrecognized logic error.\n";

     TEUCHOS_TEST_FOR_EXCEPTION_PURE_MSG(true,std::logic_error,ss.str());

  }

  // We now have a list of sideset entities, lets unwrap them and create the sideset_info!
  std::map<ParentOrdinal,std::vector<SubcellOrdinal> > owned_parent_cell_to_subcell_indexes;
  {

    // This is the subcell dimension we are trying to line up on the sideset
    const size_t face_subcell_dimension = static_cast<size_t>(mesh.getCellTopology(element_block_name)->getDimension()-1);

    // List of local subcell indexes indexed by element:
    // For example: a Tet (element) would have
    //  - 4 triangular faces (subcell_index 0-3, subcell_dimension=2)
    //  - 6 edges (subcell_index 0-5, subcell_dimension=1)
    //  - 4 vertices (subcell_index 0-3, subcell_dimension=0)
    // Another example: a Line (element) would have
    //  - 2 vertices (subcell_index 0-1, subcell_dimension=0)
    std::vector<stk::mesh::Entity> elements;
    std::vector<size_t> subcell_indexes;
    std::vector<size_t> subcell_dimensions;
    panzer_stk::workset_utils::getSideElementCascade(mesh, element_block_name, side_entities, subcell_dimensions, subcell_indexes, elements);
    const size_t num_elements = subcell_dimensions.size();

    // We need a fast lookup for maping local indexes to parent indexes
    std::unordered_map<LocalOrdinal,ParentOrdinal> local_to_parent_lookup;
    for(ParentOrdinal parent_index=0; parent_index<static_cast<ParentOrdinal>(mesh_info.local_cells.extent(0)); ++parent_index)
      local_to_parent_lookup[mesh_info.local_cells(parent_index)] = parent_index;

    // Add the subcell indexes for each element on the sideset
    // TODO: There is a lookup call here to map from local indexes to parent indexes which slows things down. Maybe there is a way around that
    for(size_t element_index=0; element_index<num_elements; ++element_index) {
      const size_t subcell_dimension = subcell_dimensions[element_index];

      // Add subcell to map
      if(subcell_dimension == face_subcell_dimension){
        const SubcellOrdinal subcell_index = static_cast<SubcellOrdinal>(subcell_indexes[element_index]);
        const LocalOrdinal element_local_index = static_cast<LocalOrdinal>(mesh.elementLocalId(elements[element_index]));

        // Look up the parent cell index using the local cell index
        const auto itr = local_to_parent_lookup.find(element_local_index);
        TEUCHOS_ASSERT(itr!= local_to_parent_lookup.end());
        const ParentOrdinal element_parent_index = itr->second;

        owned_parent_cell_to_subcell_indexes[element_parent_index].push_back(subcell_index);
      }
    }
  }

  // We now know the mapping of parent cell indexes to subcell indexes touching the sideset

  const panzer::LocalOrdinal num_owned_cells = owned_parent_cell_to_subcell_indexes.size();

  sideset_info.element_block_name = element_block_name;
  sideset_info.sideset_name = sideset_name;
  sideset_info.cell_topology = mesh.getCellTopology(element_block_name);

  sideset_info.num_owned_cells = num_owned_cells;

  struct face_t{
    face_t(const ParentOrdinal c0,
           const ParentOrdinal c1,
           const SubcellOrdinal sc0,
           const SubcellOrdinal sc1)
    {
      cell_0=c0;
      cell_1=c1;
      subcell_index_0=sc0;
      subcell_index_1=sc1;
    }
    ParentOrdinal cell_0;
    ParentOrdinal cell_1;
    SubcellOrdinal subcell_index_0;
    SubcellOrdinal subcell_index_1;
  };


  // Figure out how many cells on the other side of the sideset are ghost or virtual
  std::unordered_set<panzer::LocalOrdinal> owned_parent_cells_set, ghstd_parent_cells_set, virtual_parent_cells_set;
  std::vector<face_t> faces;
  {

    panzer::LocalOrdinal parent_virtual_cell_offset = mesh_info.num_owned_cells + mesh_info.num_ghstd_cells;
    for(const auto & local_cell_index_pair : owned_parent_cell_to_subcell_indexes){

      const ParentOrdinal parent_cell = local_cell_index_pair.first;
      const auto & subcell_indexes = local_cell_index_pair.second;

      owned_parent_cells_set.insert(parent_cell);

      for(const SubcellOrdinal & subcell_index : subcell_indexes){

        const LocalOrdinal face = mesh_info.cell_to_faces(parent_cell, subcell_index);
        const LocalOrdinal face_other_side = (mesh_info.face_to_cells(face,0) == parent_cell) ? 1 : 0;

        TEUCHOS_ASSERT(subcell_index == mesh_info.face_to_lidx(face, 1-face_other_side));

        const ParentOrdinal other_side_cell = mesh_info.face_to_cells(face, face_other_side);
        const SubcellOrdinal other_side_subcell_index = mesh_info.face_to_lidx(face, face_other_side);

        faces.push_back(face_t(parent_cell, other_side_cell, subcell_index, other_side_subcell_index));

        if(other_side_cell >= parent_virtual_cell_offset){
          virtual_parent_cells_set.insert(other_side_cell);
        } else {
          ghstd_parent_cells_set.insert(other_side_cell);
        }
      }
    }
  }

  std::vector<ParentOrdinal> all_cells;
  all_cells.insert(all_cells.end(),owned_parent_cells_set.begin(),owned_parent_cells_set.end());
  all_cells.insert(all_cells.end(),ghstd_parent_cells_set.begin(),ghstd_parent_cells_set.end());
  all_cells.insert(all_cells.end(),virtual_parent_cells_set.begin(),virtual_parent_cells_set.end());

  sideset_info.num_ghstd_cells = ghstd_parent_cells_set.size();
  sideset_info.num_virtual_cells = virtual_parent_cells_set.size();

  const LocalOrdinal num_real_cells = sideset_info.num_owned_cells + sideset_info.num_ghstd_cells;
  const LocalOrdinal num_total_cells = num_real_cells + sideset_info.num_virtual_cells;
  const LocalOrdinal num_vertices_per_cell = mesh_info.cell_vertices.extent(1);
  const LocalOrdinal num_dims = mesh_info.cell_vertices.extent(2);

  // Copy local indexes, global indexes, and cell vertices to sideset info
  {
    sideset_info.global_cells = PHX::View<panzer::GlobalOrdinal*>("global_cells", num_total_cells);
    sideset_info.local_cells = PHX::View<LocalOrdinal*>("local_cells", num_total_cells);
    sideset_info.cell_vertices = PHX::View<double***>("cell_vertices", num_total_cells, num_vertices_per_cell, num_dims);
    Kokkos::deep_copy(sideset_info.cell_vertices,0.);

    for(LocalOrdinal i=0; i<num_total_cells; ++i){
      const ParentOrdinal parent_cell = all_cells[i];
      sideset_info.local_cells(i) = mesh_info.local_cells(parent_cell);
      sideset_info.global_cells(i) = mesh_info.global_cells(parent_cell);
      for(LocalOrdinal j=0; j<num_vertices_per_cell; ++j)
        for(LocalOrdinal k=0; k<num_dims; ++k)
          sideset_info.cell_vertices(i,j,k) = mesh_info.cell_vertices(parent_cell,j,k);
    }
  }

  // Now we have to set the connectivity for the faces.

  const LocalOrdinal num_faces = faces.size();
  const LocalOrdinal num_faces_per_cell = mesh_info.cell_to_faces.extent(1);

  sideset_info.face_to_cells = PHX::View<LocalOrdinal*[2]>("face_to_cells", num_faces);
  sideset_info.face_to_lidx = PHX::View<LocalOrdinal*[2]>("face_to_lidx", num_faces);
  sideset_info.cell_to_faces = PHX::View<LocalOrdinal**>("cell_to_faces", num_total_cells, num_faces_per_cell);

  // Default the system with invalid cell index - this will be most of the entries
  Kokkos::deep_copy(sideset_info.cell_to_faces, -1);

  // Lookup for sideset cell index given parent cell index
  std::unordered_map<ParentOrdinal,ParentOrdinal> parent_to_sideset_lookup;
  for(unsigned int i=0; i<all_cells.size(); ++i)
    parent_to_sideset_lookup[all_cells[i]] = i;

  for(int face_index=0;face_index<num_faces;++face_index){
    const face_t & face = faces[face_index];
    const ParentOrdinal & parent_cell_0 = face.cell_0;
    const ParentOrdinal & parent_cell_1 = face.cell_1;

    // Convert the parent cell index into a sideset cell index
    const auto itr_0 = parent_to_sideset_lookup.find(parent_cell_0);
    TEUCHOS_ASSERT(itr_0 != parent_to_sideset_lookup.end());
    const ParentOrdinal sideset_cell_0 = itr_0->second;

    const auto itr_1 = parent_to_sideset_lookup.find(parent_cell_1);
    TEUCHOS_ASSERT(itr_1 != parent_to_sideset_lookup.end());
    const ParentOrdinal sideset_cell_1 = itr_1->second;

//    const ParentOrdinal sideset_cell_0 = std::distance(all_cells.begin(), std::find(all_cells.begin(), all_cells.end(), parent_cell_0));
//    const ParentOrdinal sideset_cell_1 = std::distance(all_cells.begin(), std::find(all_cells.begin()+num_owned_cells, all_cells.end(), parent_cell_1));

    sideset_info.face_to_cells(face_index,0) = sideset_cell_0;
    sideset_info.face_to_cells(face_index,1) = sideset_cell_1;

    sideset_info.face_to_lidx(face_index,0) = face.subcell_index_0;
    sideset_info.face_to_lidx(face_index,1) = face.subcell_index_1;

    sideset_info.cell_to_faces(sideset_cell_0,face.subcell_index_0) = face_index;
    sideset_info.cell_to_faces(sideset_cell_1,face.subcell_index_1) = face_index;

  }

}

} // namespace

Teuchos::RCP<panzer::LocalMeshInfo>
generateLocalMeshInfo(const panzer_stk::STK_Interface & mesh)
{
  using Teuchos::RCP;
  using Teuchos::rcp;

  //typedef Tpetra::CrsMatrix<int,panzer::LocalOrdinal,panzer::GlobalOrdinal> crs_type;
  typedef Tpetra::Map<panzer::LocalOrdinal,panzer::GlobalOrdinal,panzer::TpetraNodeType> map_type;
  typedef Tpetra::Import<panzer::LocalOrdinal,panzer::GlobalOrdinal,panzer::TpetraNodeType> import_type;
  //typedef Tpetra::MultiVector<double,panzer::LocalOrdinal,panzer::GlobalOrdinal> mvec_type;
  //typedef Tpetra::MultiVector<panzer::GlobalOrdinal,panzer::LocalOrdinal,panzer::GlobalOrdinal> ordmvec_type;

  auto mesh_info_rcp = Teuchos::rcp(new panzer::LocalMeshInfo);
  auto & mesh_info = *mesh_info_rcp;

  // Make sure the STK interface is valid
  TEUCHOS_ASSERT(mesh.isInitialized());

  // This is required by some of the STK stuff
  TEUCHOS_ASSERT(typeid(panzer::LocalOrdinal) == typeid(int));

  Teuchos::RCP<const Teuchos::Comm<int> > comm = mesh.getComm();

  TEUCHOS_FUNC_TIME_MONITOR_DIFF("panzer_stk::generateLocalMeshInfo",GenerateLocalMeshInfo);

  // This horrible line of code is required since the connection manager only takes rcps of a mesh
  RCP<const panzer_stk::STK_Interface> mesh_rcp = Teuchos::rcpFromRef(mesh);
  // We're allowed to do this since the connection manager only exists in this scope... even though it is also an RCP...

  // extract topology handle
  RCP<panzer::ConnManager> conn_rcp = rcp(new panzer_stk::STKConnManager(mesh_rcp));
  panzer::ConnManager & conn = *conn_rcp;

  PHX::View<panzer::GlobalOrdinal*> owned_cells, ghost_cells, virtual_cells;
  panzer::fillLocalCellIDs(comm, conn, owned_cells, ghost_cells, virtual_cells);

  // build cell maps
  /////////////////////////////////////////////////////////////////////

  RCP<map_type> owned_cell_map = rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),owned_cells,0,comm));
  RCP<map_type> ghstd_cell_map = rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),ghost_cells,0,comm));

  // build importer: cell importer, owned to ghstd
  RCP<import_type> cellimport_own2ghst = rcp(new import_type(owned_cell_map,ghstd_cell_map));

  // read all the vertices associated with these elements, get ghstd contributions
  /////////////////////////////////////////////////////////////////////

  // TODO: This all needs to be rewritten for when element blocks have different cell topologies
  std::vector<std::string> element_block_names;
  mesh.getElementBlockNames(element_block_names);

  const shards::CellTopology & cell_topology = *(mesh.getCellTopology(element_block_names[0]));

  const int space_dim = cell_topology.getDimension();
  const int vertices_per_cell = cell_topology.getVertexCount();
  const int faces_per_cell = cell_topology.getSubcellCount(space_dim-1);

  // PHX::View<double***> owned_vertices("owned_vertices",localCells.size(),vertices_per_cell,space_dim);
  Kokkos::DynRankView<double,PHX::Device> owned_vertices("owned_vertices",owned_cells.extent(0),vertices_per_cell,space_dim);
  {
    std::vector<std::size_t> localCells(owned_cells.extent(0),Teuchos::OrdinalTraits<std::size_t>::invalid());
    for(size_t i=0;i<localCells.size();i++)
      localCells[i] = i;
    mesh.getElementVerticesNoResize(localCells,owned_vertices);
  }

  // this builds a ghstd vertex array
  Kokkos::DynRankView<double,PHX::Device> ghstd_vertices = buildGhostedVertices(*cellimport_own2ghst,owned_vertices);

  // build edge to cell neighbor mapping
  //////////////////////////////////////////////////////////////////

  std::unordered_map<panzer::GlobalOrdinal,int> global_to_local;
  global_to_local[-1] = -1; // this is the "no neighbor" flag
  for(size_t i=0;i<owned_cells.extent(0);i++)
    global_to_local[owned_cells(i)] = i;
  for(size_t i=0;i<ghost_cells.extent(0);i++)
    global_to_local[ghost_cells(i)] = i+Teuchos::as<int>(owned_cells.extent(0));

  // this class comes from Mini-PIC and Matt B
  RCP<panzer::FaceToElement<panzer::LocalOrdinal,panzer::GlobalOrdinal> > faceToElement = rcp(new panzer::FaceToElement<panzer::LocalOrdinal,panzer::GlobalOrdinal>());
  faceToElement->initialize(conn);
  auto elems_by_face = faceToElement->getFaceToElementsMap();
  auto face_to_lidx  = faceToElement->getFaceToCellLocalIdxMap();

  // We also need to consider faces that connect to cells that do not exist, but are needed for boundary conditions
  // We dub them virtual cell since there should be no geometry associated with them, or topology really
  // They exist only for datastorage so that they are consistent with 'real' cells from an algorithm perspective

  // Each virtual face (face linked to a '-1' cell) requires a virtual cell (i.e. turn the '-1' into a virtual cell)
  // Virtual cells are those that do not exist but are connected to an owned cell
  // Note - in the future, ghosted cells will also need to connect to virtual cells at boundary conditions, but for the moment we will ignore this.

  // Iterate over all faces and identify the faces connected to a potential virtual cell

  const panzer::LocalOrdinal num_owned_cells = owned_cells.extent(0);
  const panzer::LocalOrdinal num_ghstd_cells = ghost_cells.extent(0);
  const panzer::LocalOrdinal num_virtual_cells = virtual_cells.extent(0);

  // total cells and faces include owned, ghosted, and virtual
  const panzer::LocalOrdinal num_real_cells = num_owned_cells + num_ghstd_cells;
  const panzer::LocalOrdinal num_total_cells = num_real_cells + num_virtual_cells;
  const panzer::LocalOrdinal num_total_faces = elems_by_face.extent(0);

  // Lookup cells connected to a face
  PHX::View<panzer::LocalOrdinal*[2]> face_to_cells = PHX::View<panzer::LocalOrdinal*[2]>("face_to_cells",num_total_faces);

  // Lookup local face indexes given cell and left/right state (0/1)
  PHX::View<panzer::LocalOrdinal*[2]> face_to_localidx = PHX::View<panzer::LocalOrdinal*[2]>("face_to_localidx",num_total_faces);

  // Lookup face index given a cell and local face index
  PHX::View<panzer::LocalOrdinal**> cell_to_face = PHX::View<panzer::LocalOrdinal**>("cell_to_face",num_total_cells,faces_per_cell);

  // initialize with negative one cells that are not associated with a face
  Kokkos::deep_copy(cell_to_face,-1);

  // Transfer information from 'faceToElement' datasets to local arrays
  {
    PANZER_FUNC_TIME_MONITOR_DIFF("Transer faceToElement to local",TransferFaceToElementLocal);

    int virtual_cell_index = num_real_cells;
    for(size_t f=0;f<elems_by_face.extent(0);f++) {

      const panzer::GlobalOrdinal global_c0 = elems_by_face(f,0);
      const panzer::GlobalOrdinal global_c1 = elems_by_face(f,1);

      // make sure that no bonus cells get in here
      TEUCHOS_ASSERT(global_to_local.find(global_c0)!=global_to_local.end());
      TEUCHOS_ASSERT(global_to_local.find(global_c1)!=global_to_local.end());

      auto c0 = global_to_local[global_c0];
      auto lidx0 = face_to_lidx(f,0);
      auto c1 = global_to_local[global_c1];
      auto lidx1 = face_to_lidx(f,1);

      // Test for virtual cells

      // Left cell
      if(c0 < 0){
        // Virtual cell - create it!
        c0 = virtual_cell_index++;

        // We need the subcell_index to line up between real and virtual cell
        // This way the face has the same geometry... though the face normal
        // will point in the wrong direction
        lidx0 = lidx1;
      }
      cell_to_face(c0,lidx0) = f;


      // Right cell
      if(c1 < 0){
        // Virtual cell - create it!
        c1 = virtual_cell_index++;

        // We need the subcell_index to line up between real and virtual cell
        // This way the face has the same geometry... though the face normal
        // will point in the wrong direction
        lidx1 = lidx0;
      }
      cell_to_face(c1,lidx1) = f;

      // Faces point from low cell index to high cell index
      if(c0<c1){
        face_to_cells(f,0) = c0;
        face_to_localidx(f,0) = lidx0;
        face_to_cells(f,1) = c1;
        face_to_localidx(f,1) = lidx1;
      } else {
        face_to_cells(f,0) = c1;
        face_to_localidx(f,0) = lidx1;
        face_to_cells(f,1) = c0;
        face_to_localidx(f,1) = lidx0;
      }

      // We should avoid having two virtual cells linked together.
      TEUCHOS_ASSERT(c0<num_real_cells or c1<num_real_cells)

    }
  }

  // at this point all the data structures have been built, so now we can "do" DG.
  // There are two of everything, an "owned" data structured corresponding to "owned"
  // cells. And a "ghstd" data structure corresponding to ghosted cells
  ////////////////////////////////////////////////////////////////////////////////////
  {
    PANZER_FUNC_TIME_MONITOR_DIFF("Assign Indices",AssignIndices);
    mesh_info.cell_to_faces           = cell_to_face;
    mesh_info.face_to_cells           = face_to_cells;      // faces
    mesh_info.face_to_lidx            = face_to_localidx;
    mesh_info.subcell_dimension       = space_dim;
    mesh_info.subcell_index           = -1;
    mesh_info.has_connectivity        = true;

    mesh_info.num_owned_cells = owned_cells.extent(0);
    mesh_info.num_ghstd_cells = ghost_cells.extent(0);
    mesh_info.num_virtual_cells = virtual_cells.extent(0);

    mesh_info.global_cells = PHX::View<panzer::GlobalOrdinal*>("global_cell_indices",num_total_cells);
    mesh_info.local_cells = PHX::View<panzer::LocalOrdinal*>("local_cell_indices",num_total_cells);

    for(int i=0;i<num_owned_cells;++i){
      mesh_info.global_cells(i) = owned_cells(i);
      mesh_info.local_cells(i) = i;
    }

    for(int i=0;i<num_ghstd_cells;++i){
      mesh_info.global_cells(i+num_owned_cells) = ghost_cells(i);
      mesh_info.local_cells(i+num_owned_cells) = i+num_owned_cells;
    }

    for(int i=0;i<num_virtual_cells;++i){
      mesh_info.global_cells(i+num_real_cells) = virtual_cells(i);
      mesh_info.local_cells(i+num_real_cells) = i+num_real_cells;
    }

    mesh_info.cell_vertices = PHX::View<double***>("cell_vertices",num_total_cells,vertices_per_cell,space_dim);

    // Initialize coordinates to zero
    Kokkos::deep_copy(mesh_info.cell_vertices, 0.);

    for(int i=0;i<num_owned_cells;++i){
      for(int j=0;j<vertices_per_cell;++j){
        for(int k=0;k<space_dim;++k){
          mesh_info.cell_vertices(i,j,k) = owned_vertices(i,j,k);
        }
      }
    }

    for(int i=0;i<num_ghstd_cells;++i){
      for(int j=0;j<vertices_per_cell;++j){
        for(int k=0;k<space_dim;++k){
          mesh_info.cell_vertices(i+num_owned_cells,j,k) = ghstd_vertices(i,j,k);
        }
      }
    }

    // This will backfire at some point, but we're going to make the virtual cell have the same geometry as the cell it interfaces with
    // This way we can define a virtual cell geometry without extruding the face outside of the domain
    {
      PANZER_FUNC_TIME_MONITOR_DIFF("Assign geometry traits",AssignGeometryTraits);
      for(int i=0;i<num_virtual_cells;++i){

        const panzer::LocalOrdinal virtual_cell = i+num_real_cells;
        bool exists = false;
        for(int local_face=0; local_face<faces_per_cell; ++local_face){
          const panzer::LocalOrdinal face = cell_to_face(virtual_cell, local_face);
          if(face >= 0){
            exists = true;
            const panzer::LocalOrdinal other_side = (face_to_cells(face, 0) == virtual_cell) ? 1 : 0;
            const panzer::LocalOrdinal real_cell = face_to_cells(face,other_side);
            TEUCHOS_ASSERT(real_cell < num_real_cells);
            for(int j=0;j<vertices_per_cell;++j){
              for(int k=0;k<space_dim;++k){
                mesh_info.cell_vertices(virtual_cell,j,k) = mesh_info.cell_vertices(real_cell,j,k);
              }
            }
            break;
          }
        }
        TEUCHOS_TEST_FOR_EXCEPT_MSG(!exists, "panzer_stk::generateLocalMeshInfo : Virtual cell is not linked to real cell");
      }
    }
  }

  // Setup element blocks and sidesets
  std::vector<std::string> sideset_names;
  mesh.getSidesetNames(sideset_names);

  for(const std::string & element_block_name : element_block_names){
    PANZER_FUNC_TIME_MONITOR_DIFF("Set up setupLocalMeshBlockInfo",SetupLocalMeshBlockInfo);
    panzer::LocalMeshBlockInfo & block_info = mesh_info.element_blocks[element_block_name];
    setupLocalMeshBlockInfo(mesh, conn, mesh_info, element_block_name, block_info);
    block_info.subcell_dimension = space_dim;
    block_info.subcell_index = -1;
    block_info.has_connectivity = true;

    // Setup sidesets
    for(const std::string & sideset_name : sideset_names){
      PANZER_FUNC_TIME_MONITOR_DIFF("Setup LocalMeshSidesetInfo",SetupLocalMeshSidesetInfo);
      panzer::LocalMeshSidesetInfo & sideset_info = mesh_info.sidesets[element_block_name][sideset_name];
      setupLocalMeshSidesetInfo(mesh, conn, mesh_info, element_block_name, sideset_name, sideset_info);
      sideset_info.subcell_dimension = space_dim;
      sideset_info.subcell_index = -1;
      sideset_info.has_connectivity = true;
    }

  }

  return mesh_info_rcp;

}

Teuchos::RCP<panzer::LocalMeshInfo>
generateLocalMeshInfo_new(const panzer_stk::STK_Interface & mesh)
{
  using Teuchos::RCP;
  using Teuchos::rcp;

  typedef Tpetra::Map<panzer::LocalOrdinal,panzer::GlobalOrdinal,panzer::TpetraNodeType> map_type;
  typedef Tpetra::Import<panzer::LocalOrdinal,panzer::GlobalOrdinal,panzer::TpetraNodeType> import_type;

  auto mesh_info_rcp = Teuchos::rcp(new panzer::LocalMeshInfo);
  auto & mesh_info = *mesh_info_rcp;

  // Make sure the STK interface is valid
  TEUCHOS_ASSERT(mesh.isInitialized());

  // This is required by some of the STK stuff
  TEUCHOS_ASSERT(typeid(panzer::LocalOrdinal) == typeid(int));

  Teuchos::RCP<const Teuchos::Comm<int> > comm = mesh.getComm();

  TEUCHOS_FUNC_TIME_MONITOR_DIFF("panzer_stk::generateLocalMeshInfo",GenerateLocalMeshInfo);

  // This horrible line of code is required since the connection manager only takes rcps of a mesh
  RCP<const panzer_stk::STK_Interface> mesh_rcp = Teuchos::rcpFromRef(mesh);
  // We're allowed to do this since the connection manager only exists in this scope... even though it is also an RCP...

  // extract topology handle
  RCP<panzer::ConnManager> conn_rcp = rcp(new panzer_stk::STKConnManager(mesh_rcp));
  panzer::ConnManager & conn = *conn_rcp;

  PHX::View<panzer::GlobalOrdinal*> owned_cells, ghost_cells;
  owned_cells = conn.getOwnedGlobalCellID(); 
  ghost_cells = conn.getGhostGlobalCellID();
	
  // ====== master mesh info ========
  mesh_info.num_owned_cells = owned_cells.extent(0);
  mesh_info.num_ghstd_cells = ghost_cells.extent(0);
  std::size_t num_total_cells = mesh_info.num_owned_cells + mesh_info.num_ghstd_cells;
  /*mesh_info.global_cells = PHX::View<panzer::GlobalOrdinal*>("global_cell_indices",num_total_cells);
  mesh_info.local_cells = PHX::View<panzer::LocalOrdinal*>("local_cell_indices",num_total_cells);
  for(int i=0;i<mesh_info.num_owned_cells;++i){
      mesh_info.global_cells(i) = owned_cells(i);
      mesh_info.local_cells(i) = i;
  }
  for(int i=0;i<mesh_info.num_ghstd_cells;++i){
      mesh_info.global_cells(i+mesh_info.num_owned_cells) = ghost_cells(i);
      mesh_info.local_cells(i+mesh_info.num_owned_cells) = i+mesh_info.num_owned_cells;
  }*/
  // ==================================

  // glocal element id -> local element id
  std::unordered_map<panzer::GlobalOrdinal,panzer::LocalOrdinal> global_to_local;
  global_to_local[-1] = -1; // this is the "no neighbor" flag
  for(size_t i=0;i<owned_cells.extent(0);i++)
    global_to_local[owned_cells(i)] = i;
  for(size_t i=0;i<ghost_cells.extent(0);i++)
    global_to_local[ghost_cells(i)] = i+Teuchos::as<int>(owned_cells.extent(0));

  // build cell maps
  /////////////////////////////////////////////////////////////////////

  RCP<map_type> owned_cell_map = rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),owned_cells,0,comm));
  RCP<map_type> ghstd_cell_map = rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),ghost_cells,0,comm));

  // build importer: cell importer, owned to ghstd
  RCP<import_type> cellimport_own2ghst = rcp(new import_type(owned_cell_map,ghstd_cell_map));
	
  ////////////////////////////////////////////
    // this class comes from Mini-PIC and Matt B
  /*RCP<panzer::FaceToElement<panzer::LocalOrdinal,panzer::GlobalOrdinal> > faceToElement = rcp(new panzer::FaceToElement<panzer::LocalOrdinal,panzer::GlobalOrdinal>());
  faceToElement->initialize(conn);
  auto elems_by_face = faceToElement->getFaceToElementsMap();
  auto face_to_lidx  = faceToElement->getFaceToCellLocalIdxMap();
  const panzer::LocalOrdinal num_total_faces = elems_by_face.extent(0);*/
	
  //stk::mesh::ElemElemGraph &elem_graph = mesh.getBulkData()->get_face_adjacent_element_graph();
  stk::mesh::EntityRank sideRank = mesh.getSideRank();

  // read all the vertices associated with these elements, get ghstd contributions
  /////////////////////////////////////////////////////////////////////

  // TODO: This all needs to be rewritten for when element blocks have different cell topologies
  std::vector<std::string> element_block_names;
  mesh.getElementBlockNames(element_block_names);
  std::vector<std::string> sideset_names;
  mesh.getSidesetNames(sideset_names);
	
  for(const std::string & element_block_name : element_block_names) {
    panzer::LocalMeshBlockInfo & block_info = mesh_info.element_blocks[element_block_name];
	block_info.element_block_name = element_block_name;
	  
    auto cell_topology = mesh.getCellTopology(element_block_name);
	block_info.cell_topology = cell_topology;
	int space_dim = cell_topology->getDimension();
    block_info.subcell_dimension = space_dim;
    int vertices_per_cell = cell_topology->getVertexCount();
    int faces_per_cell = cell_topology->getSubcellCount(space_dim-1);
	  
	std::vector<stk::mesh::Entity> my_elements, ghost_elements;
    mesh.getMyElements( element_block_name, my_elements );
    mesh.getNeighborElements( element_block_name, ghost_elements );
    block_info.num_owned_cells = my_elements.size();
    block_info.num_ghstd_cells = ghost_elements.size();
	block_info.num_virtual_cells = 0;   // for consitence with old design
    num_total_cells = block_info.num_owned_cells + block_info.num_ghstd_cells;
	  
	block_info.global_cells = PHX::View<panzer::GlobalOrdinal*>("global_cells", num_total_cells);
    block_info.local_cells = PHX::View<panzer::LocalOrdinal*>("local_cells", num_total_cells);
	  
    Kokkos::DynRankView<double,PHX::Device> owned_vertices("owned_vertices",owned_cells.extent(0),vertices_per_cell,space_dim);
    mesh.getElementVerticesNoResize(my_elements,owned_vertices);
	Kokkos::DynRankView<double,PHX::Device> ghost_vertices = buildGhostedVertices(*cellimport_own2ghst,owned_vertices);
	  
	block_info.cell_vertices = PHX::View<double***>("cell_vertices", num_total_cells, vertices_per_cell, space_dim);
    for(int i=0;i<block_info.num_owned_cells;++i) {
      for(int j=0;j<vertices_per_cell;++j) {
        for(int k=0;k<space_dim;++k){
          block_info.cell_vertices(i,j,k) = owned_vertices(i,j,k);
        }
      }
	  block_info.global_cells(i) = owned_cells(i);
      block_info.local_cells(i) = i;
    }

    for(int i=0;i<block_info.num_ghstd_cells;++i) {
      for(int j=0;j<vertices_per_cell;++j) {
        for(int k=0;k<space_dim;++k){
          block_info.cell_vertices(i+block_info.num_owned_cells,j,k) = ghost_vertices(i,j,k);
        }
      }
	  block_info.global_cells(i+block_info.num_owned_cells) = ghost_cells(i);
      block_info.local_cells(i+block_info.num_owned_cells) = i+block_info.num_owned_cells;
    }
	  
	// Faces
	//if( mesh.getBulkData()->has_face_adjacent_element_graph() ) {

	// cell_to_faces : ELement LID -> faces GIDs
	block_info.cell_to_faces = PHX::View<panzer::LocalOrdinal**>("cell_to_face",num_total_cells,faces_per_cell);
	std::set<stk::mesh::Entity> faces;
	for( int i=0;i<block_info.num_owned_cells;++i ) {
		const stk::mesh::Entity* faceElement = mesh.getBulkData()->begin(my_elements[i], sideRank);
		unsigned numSides = mesh.getBulkData()->num_connectivity( my_elements[i], sideRank);
		for (unsigned j = 0; j < numSides; ++j)
        {
            block_info.cell_to_faces(i,j) = mesh.getBulkData()->identifier( faceElement[j] );
			faces.insert( faceElement[j] );
        }
	}
	for( int i=0;i<block_info.num_ghstd_cells;++i ) {
		const stk::mesh::Entity* faceElement = mesh.getBulkData()->begin(ghost_elements[i], sideRank);
		unsigned numSides = mesh.getBulkData()->num_sides(*faceElement);
		for (unsigned j = 0; j < numSides; ++j)
        {
            block_info.cell_to_faces(i+block_info.num_owned_cells,j) = mesh.getBulkData()->identifier( faceElement[j] );
			faces.insert( faceElement[j] );
        }
	}
	std::cout << "Number of owned and ghost element=" << block_info.num_owned_cells  << "," << block_info.num_ghstd_cells
		<< " with number of faces= " << faces.size() << " in rank" << comm->getRank() << std::endl;

	// face_to_cells : Face GIDs => Element[0] LID + Element[1] LID
	block_info.face_to_cells = PHX::View<panzer::LocalOrdinal*[2]>("face_to_cells",faces.size());
    std::unordered_map<panzer::GlobalOrdinal,panzer::LocalOrdinal> face_g2l;
	//stk::mesh::EntityRank eleRank = sideRank+1:
	std::size_t numFace = 0;
    for( const auto face: faces ) {
		const stk::mesh::Entity* elements = mesh.getBulkData()->begin_elements(face);
		unsigned numEles = mesh.getBulkData()->num_elements( face );
 	//	std::cout << mesh.getBulkData()->identifier(face) << ",a  " << numEles << std::endl;
		panzer::GlobalOrdinal global_c0 = mesh.getBulkData()->identifier( elements[0] ) -1;
		TEUCHOS_ASSERT(global_to_local.find(global_c0)!=global_to_local.end());
		panzer::LocalOrdinal c0 = global_to_local[global_c0];
		block_info.face_to_cells( numFace, 0 ) = c0;
	//	std::cout <<    "  :,a  " << numFace << "," << c0 << std::endl;
		panzer::GlobalOrdinal global_c1 = -1;
		panzer::LocalOrdinal c1 = -1;
		if( numEles>=2 ) {
        	global_c1 = mesh.getBulkData()->identifier( elements[1] ) -1;
      		TEUCHOS_ASSERT(global_to_local.find(global_c1)!=global_to_local.end());
			c1 = global_to_local[global_c1];
		}
		block_info.face_to_cells( numFace, 1 ) = c1;
	//	std::cout <<    "  :,a  " << global_c1 << "," << c1 << std::endl;
		face_g2l[ mesh.getBulkData()->identifier(face)] =numFace;
		++numFace;		
	}
	  
	// cell_to_faces : ELement LID -> faces LIDs
	// face_to_lidx: FaceLIDs (FaceGIDs) -> side index of parent cell
	std::vector<panzer::LocalOrdinal> counter(faces.size(),0);
	block_info.face_to_lidx = PHX::View<panzer::LocalOrdinal*[2]>("face_to_localidx",faces.size());
	Kokkos::deep_copy(block_info.face_to_lidx,-1);
	for( int i=0;i<block_info.num_owned_cells;++i ) {
		const stk::mesh::Entity* faceElement = mesh.getBulkData()->begin(my_elements[i], sideRank);
		unsigned numSides = mesh.getBulkData()->num_connectivity( my_elements[i], sideRank);
		for (unsigned j = 0; j < numSides; ++j)
        {
            panzer::GlobalOrdinal global_c = block_info.cell_to_faces(i,j);
			TEUCHOS_ASSERT(face_g2l.find(global_c)!=face_g2l.end());
			auto lid = face_g2l[global_c];
			block_info.face_to_lidx( lid, counter[lid] ) = j;
			//std::cout << global_c << ", " << lid << ", aa:" << counter[lid] << ", " << block_info.face_to_lidx( lid, counter[lid] ) << std::endl;
			++(counter[lid]);
			block_info.cell_to_faces(i,j) = lid;
        }
	}

    block_info.subcell_index = -1;
    block_info.has_connectivity = true;

	std::vector<stk::mesh::Entity> side_entities;
	for(const std::string & sideset_name : sideset_names){
      panzer::LocalMeshSidesetInfo & sideset_info = mesh_info.sidesets[element_block_name][sideset_name];
	  mesh.getAllSides(sideset_name, element_block_name, side_entities);
	  if( side_entities.empty() ) continue;
      //setupLocalMeshSidesetInfo(mesh, conn, mesh_info, element_block_name, sideset_name, sideset_info);
      //sideset_info.subcell_dimension = space_dim;
      //sideset_info.subcell_index = -1;
      //sideset_info.has_connectivity = true;
    }
 }

  return mesh_info_rcp;

}


}
