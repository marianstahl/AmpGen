#include <chrono>
#include <ctime>
#include <iostream>
#include <map>
#include <ratio>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "THStack.h"
#include "TCanvas.h"

#include "AmpGen/Chi2Estimator.h"
#include "AmpGen/ErrorPropagator.h"
#include "AmpGen/EventList.h"
#include "AmpGen/EventType.h"
#include "AmpGen/Factory.h"
#include "AmpGen/RecursivePhaseSpace.h"
#include "AmpGen/FitResult.h"
#include "AmpGen/IExtendLikelihood.h"
#include "AmpGen/Minimiser.h"
#include "AmpGen/MinuitParameter.h"
#include "AmpGen/MinuitParameterSet.h"
#include "AmpGen/MsgService.h"
#include "AmpGen/NamedParameter.h"
#include "AmpGen/SumPDF.h"
#include "AmpGen/ThreeBodyCalculators.h"
#include "AmpGen/Utilities.h"
#include "AmpGen/Generator.h"
#include "AmpGen/PolarisedSum.h"
#include "AmpGen/Plots.h"
#include "AmpGen/Kinematics.h"

#ifdef _OPENMP
  #include <omp.h>
  #include <thread>
#endif

using namespace AmpGen;

double decayAngle0( const Event& evt)
{
  TLorentzVector p1 = pFromEvent(evt,0);
  return 180. * p1.Vect().Theta() / M_PI;
}

double decayAngle1( const Event& evt)
{
  TLorentzVector p1 = pFromEvent(evt,0);
  return 180. * p1.Vect().Phi() / M_PI;
}

double decayAngle2( const Event& evt)
{
  TLorentzVector pP  = pFromEvent(evt,0);
  TLorentzVector pH1 = pFromEvent(evt,1);
  TLorentzVector pH2 = pFromEvent(evt,2);
  TVector3 z   = TVector3(0,0,1);
  TVector3 qA  = ( z.Cross( pP.Vect() )).Unit();
  TVector3 di  = pH1.Vect().Cross( pH2.Vect() );
  return 180 * di.Angle(qA) / M_PI;
}

/*
unsigned int count_amplitudes( const AmpGen::MinuitParameterSet& mps )
{
  unsigned int counter = 0;
  for ( auto param = mps.cbegin(); param != mps.cend(); ++param ) {
    if ( ( *param )->name().find( "_Re" ) != std::string::npos ) counter++;
  }
  return counter;
}
*/

void randomizeStartingPoint( MinuitParameterSet& mps, TRandom3& rand)
{
  for (auto& param : mps) {
    if ( ! param->isFree() || param->name() == "Px" || param->name() == "Py" || param->name() == "Pz" ) continue;
    double min = param->minInit();
    double max = param->maxInit();
    double new_value = rand.Uniform(param->mean()-param->stepInit(),param->mean()+param->stepInit());
    if( min != 0 && max != 0 )
      new_value = rand.Uniform(min,max);
    param->setInit( new_value );
    param->setCurrentFitVal( new_value );
    INFO( param->name() << "  = " << param->mean() << " " << param->stepInit() );
  }
}

template <typename PDF>
FitResult* doFit( PDF&& pdf, EventList& data, EventList& mc, MinuitParameterSet& MPS )
{
  auto time_wall = std::chrono::high_resolution_clock::now();
  auto time      = std::clock();

  pdf.setEvents( data );
  //for_each( pdf.pdfs(), [](auto& f ){ f.prepare(); } );

  /* Minimiser is a general interface to Minuit1/Minuit2,
     that is constructed from an object that defines an operator() that returns a double
     (i.e. the likielihood, and a set of MinuitParameters. */
  Minimiser mini( pdf, &MPS );
  mini.doFit();
  FitResult* fr = new FitResult(mini);

  /* Make the plots for the different components in the PDF, i.e. the signal and backgrounds.
     The structure assumed the PDF is some SumPDF<T1,T2,...>. */
  unsigned int counter = 1;
  for_each(pdf.pdfs(), [&]( auto& f ){
    auto mc_plot3 = mc.makeDefaultProjections(WeightFunction(f), Prefix("Model_cat"+std::to_string(counter)));
    for( auto& plot : mc_plot3 )
    {
      plot->Scale( ( data.integral() * f.getWeight() ) / plot->Integral() );
      plot->Write();
    }
    mc.transform([&f](auto& mcevt){mcevt.setWeight(f.prob(mcevt)*mcevt.weight()/mcevt.genPdf());}).tree(counter>1?"MCt"+std::to_string(counter):"MCt")->Write();
    data.tree(counter>1?"t"+std::to_string(counter):"t")->Write();
    counter++;
  } );
  /* Estimate the chi2 using an adaptive / decision tree based binning,
     down to a minimum bin population of 15, and add it to the output.*/
  if(data.eventType().size() < 5){
    Chi2Estimator chi2( data, mc, pdf, 15 );
    //chi2.writeBinningToFile("chi2_binning.txt");
    fr->addChi2( chi2.chi2(), chi2.nBins() );
  }

  auto twall_end  = std::chrono::high_resolution_clock::now();
  double time_cpu = ( std::clock() - time ) / (double)CLOCKS_PER_SEC;
  double tWall    = std::chrono::duration<double, std::milli>( twall_end - time_wall ).count();
  INFO( "Wall time = " << tWall / 1000. );
  INFO( "CPU  time = " << time_cpu );
  fr->print();
  return fr;
}

void invertParity( Event& event, const size_t& nParticles=0)
{
  for( size_t i = 0 ; i < nParticles; ++i )
  {
    event[4*i + 0] = -event[4*i+0];
    event[4*i + 1] = -event[4*i+1];
    event[4*i + 2] = -event[4*i+2];
  }
}

int main( int argc, char* argv[] )
{
  gErrorIgnoreLevel = 1001;

  OptionsParser::setArgs( argc, argv );

  const std::vector<std::string> dataFile = NamedParameter<std::string>("DataSample","").getVector();
  const std::string mcFile                = NamedParameter<std::string>("SgIntegratorFname","");
  const std::string logFile               = NamedParameter<std::string>("LogFile","Fitter.log");
  const std::string plotFile              = NamedParameter<std::string>("Plots","plots.root");
  const std::string prefix                = NamedParameter<std::string>("PlotPrefix","");
  const std::string idbranch              = NamedParameter<std::string>("IDBranch","");
  const std::string weight_branch         = NamedParameter<std::string>("WeightBranch","weight","Name of branch containing event weights.");

  const auto nev_MC = NamedParameter<int>("NEventsMC", 8e6, "Number of MC events for normalization.");
  auto bNames = NamedParameter<std::string>("Branches", std::vector<std::string>()
              ,"List of branch names, assumed to be \033[3m daughter1_px ... daughter1_E, daughter2_px ... \033[0m" ).getVector();
  auto pNames = NamedParameter<std::string>("EventType" , ""
              , "EventType to fit, in the format: \033[3m parent daughter1 daughter2 ... \033[0m" ).getVector();

#ifdef _OPENMP
  unsigned int concurentThreadsSupported = std::thread::hardware_concurrency();
  unsigned int nThreads                  = NamedParameter<unsigned int>( "nCores", concurentThreadsSupported );
  omp_set_num_threads( nThreads );
  INFO( "Setting " << nThreads << " fixed threads for OpenMP" );
  omp_set_dynamic( 0 );
#endif


  /* A MinuitParameterSet is (unsurprisingly) a set of fit parameters, and can be loaded from
     the parsed options. For historical reasons, this is referred to as loading it from a "Stream" */
  MinuitParameterSet MPS;
  MPS.loadFromStream();
  TRandom3 rndm = TRandom3( NamedParameter<unsigned int>("Seed", 0 ) ) ;
  if( NamedParameter<bool>("RandomizeStartingPoint",false) ) randomizeStartingPoint(MPS,rndm );

  /* An EventType specifies the initial and final state particles as a vector that will be described by the fit.
     It is typically loaded from the interface parameter EventType. */
  EventType evtType(pNames);

  /* A CoherentSum is the typical amplitude to be used, that is some sum over quasi two-body contributions
     weighted by an appropriate complex amplitude. The CoherentSum is generated from the couplings described
     by a set of parameters (in a MinuitParameterSet), and an EventType, which matches these parameters
     to a given final state and a set of data. A common set of rules can be matched to multiple final states,
     i.e. to facilitate the analysis of coupled channels. */
  PolarisedSum sig(evtType, MPS);

  /* Events are read in from ROOT files. If only the filename and the event type are specified,
     the file is assumed to be in the specific format that is defined by the event type,
     unless the branches to load are specified in the user options */
  EventType evtType_primed = evtType;
  if(!idbranch.empty())
    evtType_primed.extendEventType(idbranch);
  EventList events(dataFile, evtType_primed, Branches(bNames), GetGenPdf(false), WeightBranch(weight_branch) );

  /* Generate events to normalise the PDF with. This can also be loaded from a file,
     which will be the case when efficiency variations are included. */
  EventList eventsMC = Generator<RecursivePhaseSpace>(sig.matrixElements()[0].decayTree.quasiStableTree(), evtType, &rndm).generate(nev_MC);
  //EventList eventsMC = Generator<>(evtType, &rndm).generate(nev_MC);

  /* Transform data if we have an ID brach. That branch indicates that we operate on a sample with particles+antiparticles mixed.
     The transformation also includes boosting to the restframe of the head of the decay chain.
     TODO: There might be situations where you want to separate both transformations */
  if(!idbranch.empty()){
    auto frame_transform = [&evtType](auto& event){
      TVector3 pBeam(0,0,1);
      if( event[event.size()-1] < 0 ){
        invertParity( event, 3);
        pBeam = -pBeam;
      }
      std::vector<unsigned> daughters_as_ints(evtType.size());
      std::iota (daughters_as_ints.begin(), daughters_as_ints.end(), 0u);
      TLorentzVector pP = pFromEvent(event,daughters_as_ints);
      //if( pP.P() < 10e-5) return;
      TVector3 pZ = pP.Vect();
      rotateBasis( event, (pBeam.Cross(pZ) ).Cross(pZ), pBeam.Cross(pZ), pZ );
      boost( event, {0, 0, -1}, pP.P()/pP.E() );
    };

    for( auto& event : events )
      if( event[event.size()-1] < 0 ){
        event.print();
        break;
      }
    events.transform( frame_transform );
    for( auto& event : events )
      if( event[event.size()-1] < 0 ){
        event.print();
        break;
      }
  }
  sig.setMC(eventsMC);
  sig.setEvents(events);

  TFile* output = TFile::Open( plotFile.c_str(), "RECREATE" );
  output->cd();
  auto fr = doFit(make_pdf(sig), events, eventsMC, MPS);

  auto ff = sig.fitFractions( fr->getErrorPropagator() );
  fr->addFractions(ff);
  fr->writeToFile( logFile );
  output->cd();

  std::vector<Projection> projections = evtType.defaultProjections(100);
  projections.emplace_back( decayAngle0, "angle0", "\\theta_{z}"       ,100,0,180.   , "^{\\mathrm{o}}");
  projections.emplace_back( decayAngle1, "angle1", "\\phi_{x}"         ,100,-180,180 , "^{\\mathrm{o}}");
  projections.emplace_back( decayAngle2, "angle2", "\\phi_{h^{+}h^{-}}",100,0,180    , "^{\\mathrm{o}}");
  projections.emplace_back( HelicityCosine({0},{1},{1,2}), "hCos1", "\\cos\\left(\\theta\\right)",100,-1.,1);
  projections.emplace_back( HelicityCosine({0},{2},{0,1}), "hCos2", "\\cos\\left(\\theta^{\\prime}\\right)",100,-1.,1);
  auto data_plots = events.makeProjections( projections );
  std::function<double(const Event&)> FCN_sig = [&](const Event& evt){ return sig.prob_unnormalised(evt) ; };
  //std::function<double(const Event&)> FCN_bkg = [&](const Event& evt){ return bkg.prob_unnormalised(evt) ; };
  auto sig_plots      = eventsMC.makeProjections(projections, WeightFunction(FCN_sig), Prefix("MC_sig"));
  //auto bkg_plots      = eventsMC.makeProjections(projections, WeightFunction(FCN_bkg), Prefix("MC_bkg"));
  auto plots_noWeight = eventsMC.makeProjections(projections, Prefix("MC_noweight") );

  for ( unsigned int i = 0 ; i < data_plots.size() ; ++i )
  {
    //std::string figureName = prefix + std::string( data_plots[i]->GetName() ) + ".pdf";
    //INFO("Writing plot → " << figureName);
    auto& plot = data_plots[i];
    auto&   mc = sig_plots[i];
    //auto&  bkg = bkg_plots[i];
    mc->Scale( plot->Integral() / mc->Integral() );
    //bkg->Scale( MPS["fBkg"]->mean() * plot->Integral() / bkg->Integral() );
    plot->Draw("E");
    THStack* hs = new THStack();
    //if(hasBackground) hs->Add( bkg, "C HIST");
    hs->Add( mc, "C HIST");
    mc->SetLineColor(658);
    //bkg->SetFillColorAlpha(ANAGreen,0.5);
    //bkg->SetLineColor(ANAGreen+1);
    hs->Draw("same");
    hs->SetMinimum(0);
    plot->Draw("E same");
    gPad->RedrawAxis();
    //LatexViewer().view( (TCanvas*)gPad, figureName);
    plot->Write();
    mc->Write();
    //bkg->Write();
    plots_noWeight[i]->Write();
  }
  output->Close();
  return 0;
}
